-- ══════════════════════════════════════════════════════════════
-- KURIA — Schéma complet Supabase (PostgreSQL)
-- 
-- 11 tables :
--   companies        → Profils d'entreprises
--   agent_configs    → Configuration des agents par client
--   events_raw       → Tous les événements entrants
--   scan_history     → Historique des scans (Clarity Score)
--   decisions        → Décisions produites par les LLMs
--   actions          → Actions à exécuter / en attente
--   action_logs      → Audit trail complet
--   runs             → Résultats d'exécution des cycles
--   reports          → Rapports hebdomadaires
--   signals          → Signaux inter-agents
--   metrics_history  → Snapshots de métriques dans le temps
--
-- Convention :
--   - UUIDs partout
--   - created_at + updated_at sur chaque table
--   - company_id comme FK partout
--   - JSONB pour les données flexibles
--   - Indexes sur les colonnes filtrées fréquemment
-- ══════════════════════════════════════════════════════════════


-- ──────────────────────────────────────────────────────
-- EXTENSIONS
-- ──────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Pour la recherche fuzzy


-- ──────────────────────────────────────────────────────
-- 1. COMPANIES
-- ──────────────────────────────────────────────────────

CREATE TABLE companies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    industry        TEXT DEFAULT '',
    size            TEXT DEFAULT 'small' CHECK (size IN ('micro', 'small', 'medium')),
    growth_stage    TEXT DEFAULT 'scaling' CHECK (growth_stage IN ('startup', 'scaling', 'mature')),
    
    -- Outils connectés
    connected_tools JSONB DEFAULT '[]'::JSONB,
    -- Ex: ["hubspot", "quickbooks", "gmail"]
    
    -- Contact principal
    primary_contact_name  TEXT DEFAULT '',
    primary_contact_email TEXT DEFAULT '',
    
    -- Clarity Score actuel
    clarity_score         REAL DEFAULT 0 CHECK (clarity_score >= 0 AND clarity_score <= 100),
    clarity_score_prev    REAL DEFAULT NULL,
    
    -- Méta
    onboarded_at    TIMESTAMPTZ DEFAULT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_companies_active ON companies (is_active) WHERE is_active = TRUE;
CREATE INDEX idx_companies_name ON companies USING gin (name gin_trgm_ops);


-- ──────────────────────────────────────────────────────
-- 2. AGENT CONFIGS
-- ──────────────────────────────────────────────────────

CREATE TABLE agent_configs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    -- Config complète sérialisée (AgentConfigSet)
    config_data     JSONB NOT NULL DEFAULT '{}'::JSONB,
    
    -- Agents activés (dénormalisé pour les queries rapides)
    revenue_velocity_enabled       BOOLEAN DEFAULT FALSE,
    process_clarity_enabled        BOOLEAN DEFAULT FALSE,
    cash_predictability_enabled    BOOLEAN DEFAULT FALSE,
    acquisition_efficiency_enabled BOOLEAN DEFAULT FALSE,
    
    -- Adaptations
    adaptation_count  INTEGER DEFAULT 0,
    last_adapted_at   TIMESTAMPTZ DEFAULT NULL,
    
    -- Méta
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (company_id)
);

CREATE INDEX idx_agent_configs_company ON agent_configs (company_id);


-- ──────────────────────────────────────────────────────
-- 3. EVENTS RAW — Event Layer
-- ──────────────────────────────────────────────────────

CREATE TABLE events_raw (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    event_type      TEXT NOT NULL,
    -- Ex: "deal.updated", "invoice.paid", "task.completed"
    
    source          TEXT NOT NULL,
    -- Ex: "hubspot", "stripe", "gmail", "slack"
    
    actor           TEXT DEFAULT '',
    -- Qui a déclenché : email, "system", "kuria_agent"
    
    payload         JSONB NOT NULL DEFAULT '{}'::JSONB,
    metadata        JSONB DEFAULT '{}'::JSONB,
    
    -- Processing
    processed       BOOLEAN DEFAULT FALSE,
    processed_at    TIMESTAMPTZ DEFAULT NULL,
    processed_by    TEXT DEFAULT NULL,
    -- Ex: "revenue_velocity", "process_clarity"
    
    -- Méta
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_company ON events_raw (company_id);
CREATE INDEX idx_events_type ON events_raw (event_type);
CREATE INDEX idx_events_source ON events_raw (source);
CREATE INDEX idx_events_unprocessed ON events_raw (company_id, processed) WHERE processed = FALSE;
CREATE INDEX idx_events_timestamp ON events_raw (timestamp DESC);

-- Partition par mois pour la performance (optionnel, activer quand > 1M rows)
-- CREATE TABLE events_raw_y2025m01 PARTITION OF events_raw
--     FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');


-- ──────────────────────────────────────────────────────
-- 4. SCAN HISTORY
-- ──────────────────────────────────────────────────────

CREATE TABLE scan_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    scan_number     INTEGER NOT NULL DEFAULT 1,
    scan_mode       TEXT NOT NULL DEFAULT 'delta' CHECK (scan_mode IN ('initial', 'delta', 'deep')),
    
    -- Clarity Score
    clarity_score   REAL CHECK (clarity_score >= 0 AND clarity_score <= 100),
    clarity_details JSONB DEFAULT '{}'::JSONB,
    -- Contient : machine_readability, structural_compatibility, sous-scores
    
    -- Résultats
    top_frictions       JSONB DEFAULT '[]'::JSONB,
    agent_recommendations JSONB DEFAULT '{}'::JSONB,
    data_portrait       JSONB DEFAULT '{}'::JSONB,
    estimated_annual_waste REAL DEFAULT 0,
    progression         JSONB DEFAULT '{}'::JSONB,
    next_scan_focus     TEXT DEFAULT '',
    
    -- Lien vers la decision
    decision_id     UUID DEFAULT NULL,
    
    -- Méta
    scanned_at      TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scans_company ON scan_history (company_id);
CREATE INDEX idx_scans_company_date ON scan_history (company_id, scanned_at DESC);
CREATE INDEX idx_scans_mode ON scan_history (scan_mode);


-- ──────────────────────────────────────────────────────
-- 5. DECISIONS — Decision Engine
-- ──────────────────────────────────────────────────────

CREATE TABLE decisions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    agent_type      TEXT NOT NULL,
    decision_type   TEXT NOT NULL,
    confidence      REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reasoning       TEXT DEFAULT '',
    risk_level      TEXT NOT NULL DEFAULT 'A' CHECK (risk_level IN ('A', 'B', 'C')),
    
    -- Actions demandées (array JSON)
    actions         JSONB DEFAULT '[]'::JSONB,
    
    -- Metadata (KPIs, métriques produites par le LLM)
    metadata        JSONB DEFAULT '{}'::JSONB,
    
    -- State
    snapshot_id     TEXT DEFAULT '',
    validated       BOOLEAN DEFAULT FALSE,
    validation_errors JSONB DEFAULT '[]'::JSONB,
    executed        BOOLEAN DEFAULT FALSE,
    executed_at     TIMESTAMPTZ DEFAULT NULL,
    
    -- Méta
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_decisions_company ON decisions (company_id);
CREATE INDEX idx_decisions_agent ON decisions (agent_type);
CREATE INDEX idx_decisions_type ON decisions (decision_type);
CREATE INDEX idx_decisions_company_date ON decisions (company_id, created_at DESC);
CREATE INDEX idx_decisions_pending ON decisions (company_id, executed) WHERE executed = FALSE;


-- ──────────────────────────────────────────────────────
-- 6. ACTIONS — Action Executor + Pending Approvals
-- ──────────────────────────────────────────────────────

CREATE TABLE actions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id     UUID REFERENCES decisions(id) ON DELETE SET NULL,
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    agent_type      TEXT NOT NULL,
    action          TEXT NOT NULL,
    -- Ex: "update_deal_stage", "send_slack", "create_sop"
    
    target          TEXT DEFAULT '',
    parameters      JSONB DEFAULT '{}'::JSONB,
    
    -- Safety
    risk_level      TEXT NOT NULL DEFAULT 'A' CHECK (risk_level IN ('A', 'B', 'C')),
    confidence      REAL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Status
    status          TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN (
            'pending', 'pending_approval', 'approved', 'executing',
            'completed', 'failed', 'expired', 'rejected', 'briefing_only'
        )),
    
    -- Approval
    expires_at      TIMESTAMPTZ DEFAULT NULL,
    approved_by     TEXT DEFAULT NULL,
    approved_at     TIMESTAMPTZ DEFAULT NULL,
    
    -- Result
    result          JSONB DEFAULT '{}'::JSONB,
    error           TEXT DEFAULT NULL,
    executed_at     TIMESTAMPTZ DEFAULT NULL,
    
    -- Méta
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_actions_company ON actions (company_id);
CREATE INDEX idx_actions_status ON actions (status);
CREATE INDEX idx_actions_pending ON actions (company_id, status) 
    WHERE status IN ('pending', 'pending_approval');
CREATE INDEX idx_actions_decision ON actions (decision_id);
CREATE INDEX idx_actions_expiry ON actions (expires_at) 
    WHERE status = 'pending_approval' AND expires_at IS NOT NULL;


-- ──────────────────────────────────────────────────────
-- 7. ACTION LOGS — Audit Trail
-- ──────────────────────────────────────────────────────

CREATE TABLE action_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    agent_type      TEXT DEFAULT '',
    action_id       UUID DEFAULT NULL,
    decision_id     UUID DEFAULT NULL,
    
    event_type      TEXT NOT NULL DEFAULT '',
    -- Ex: "agent_run", "action.update_deal_stage", "action.rejected"
    
    description     TEXT DEFAULT '',
    
    -- Input / Output
    input_snapshot  JSONB DEFAULT '{}'::JSONB,
    output_decision JSONB DEFAULT '{}'::JSONB,
    output_result   JSONB DEFAULT '{}'::JSONB,
    
    -- LLM metrics
    llm_provider    TEXT DEFAULT '',
    llm_model       TEXT DEFAULT '',
    llm_tokens      INTEGER DEFAULT 0,
    llm_cost_usd    REAL DEFAULT 0,
    latency_ms      REAL DEFAULT 0,
    
    -- Status
    success         BOOLEAN DEFAULT TRUE,
    error           TEXT DEFAULT NULL,
    
    -- Méta
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_logs_company ON action_logs (company_id);
CREATE INDEX idx_logs_agent ON action_logs (agent_type);
CREATE INDEX idx_logs_timestamp ON action_logs (timestamp DESC);
CREATE INDEX idx_logs_errors ON action_logs (company_id, success) WHERE success = FALSE;


-- ──────────────────────────────────────────────────────
-- 8. RUNS — Cycle d'orchestration
-- ──────────────────────────────────────────────────────

CREATE TABLE runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    run_type        TEXT NOT NULL DEFAULT 'daily'
        CHECK (run_type IN ('daily', 'weekly', 'on_demand', 'initial_scan')),
    
    -- Résultat par agent
    agent_runs      JSONB DEFAULT '[]'::JSONB,
    -- Array of: {agent_type, status, kpi_value, frictions_detected, duration_seconds}
    
    -- Agrégats
    total_frictions INTEGER DEFAULT 0,
    signals_emitted INTEGER DEFAULT 0,
    all_success     BOOLEAN DEFAULT TRUE,
    
    -- Timing
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ DEFAULT NULL,
    duration_seconds REAL DEFAULT 0,
    
    -- Méta
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_runs_company ON runs (company_id);
CREATE INDEX idx_runs_company_date ON runs (company_id, created_at DESC);
CREATE INDEX idx_runs_type ON runs (run_type);


-- ──────────────────────────────────────────────────────
-- 9. REPORTS — Rapports hebdomadaires
-- ──────────────────────────────────────────────────────

CREATE TABLE reports (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    period_start    DATE NOT NULL,
    period_end      DATE NOT NULL,
    
    -- Contenu complet (WeeklyReport sérialisé)
    report_data     JSONB NOT NULL DEFAULT '{}'::JSONB,
    
    -- Dénormalisé pour queries rapides
    clarity_score       REAL DEFAULT NULL,
    attention_level     TEXT DEFAULT 'info' CHECK (attention_level IN ('critical', 'warning', 'info', 'success')),
    key_recommendation  TEXT DEFAULT '',
    total_decisions     INTEGER DEFAULT 0,
    total_actions       INTEGER DEFAULT 0,
    llm_cost_usd        REAL DEFAULT 0,
    
    -- Delivery
    email_sent      BOOLEAN DEFAULT FALSE,
    email_sent_at   TIMESTAMPTZ DEFAULT NULL,
    slack_sent      BOOLEAN DEFAULT FALSE,
    
    -- Méta
    generated_at    TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reports_company ON reports (company_id);
CREATE INDEX idx_reports_company_date ON reports (company_id, period_end DESC);


-- ──────────────────────────────────────────────────────
-- 10. SIGNALS — Inter-agent communication
-- ──────────────────────────────────────────────────────

CREATE TABLE signals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    signal_type     TEXT NOT NULL,
    priority        TEXT DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    source_agent    TEXT NOT NULL,
    target_agents   JSONB DEFAULT '[]'::JSONB,
    
    payload         JSONB DEFAULT '{}'::JSONB,
    message         TEXT DEFAULT '',
    
    -- Processing
    processed       BOOLEAN DEFAULT FALSE,
    processed_at    TIMESTAMPTZ DEFAULT NULL,
    
    -- Méta
    emitted_at      TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_company ON signals (company_id);
CREATE INDEX idx_signals_unprocessed ON signals (company_id, processed) WHERE processed = FALSE;
CREATE INDEX idx_signals_source ON signals (source_agent);


-- ──────────────────────────────────────────────────────
-- 11. METRICS HISTORY — KPI snapshots
-- ──────────────────────────────────────────────────────

CREATE TABLE metrics_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    agent_type      TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    -- Ex: "cycle_time_days", "blended_cac", "runway_months"
    
    value           REAL NOT NULL,
    unit            TEXT DEFAULT '',
    health          TEXT DEFAULT 'unknown' CHECK (health IN ('healthy', 'warning', 'critical', 'unknown')),
    
    details         JSONB DEFAULT '{}'::JSONB,
    
    -- Méta
    recorded_at     TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metrics_company ON metrics_history (company_id);
CREATE INDEX idx_metrics_lookup ON metrics_history (company_id, metric_name, recorded_at DESC);
CREATE INDEX idx_metrics_agent ON metrics_history (agent_type);


-- ══════════════════════════════════════════════════════════════
-- FUNCTIONS
-- ══════════════════════════════════════════════════════════════


-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_companies_updated
    BEFORE UPDATE ON companies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_agent_configs_updated
    BEFORE UPDATE ON agent_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();


-- Expire pending actions
CREATE OR REPLACE FUNCTION expire_pending_actions()
RETURNS void AS $$
BEGIN
    UPDATE actions
    SET status = 'expired'
    WHERE status = 'pending_approval'
      AND expires_at IS NOT NULL
      AND expires_at < NOW();
END;
$$ LANGUAGE plpgsql;


-- Update company clarity score from latest scan
CREATE OR REPLACE FUNCTION update_company_clarity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE companies
    SET clarity_score_prev = clarity_score,
        clarity_score = NEW.clarity_score,
        updated_at = NOW()
    WHERE id = NEW.company_id
      AND NEW.clarity_score IS NOT NULL;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_scan_updates_company
    AFTER INSERT ON scan_history
    FOR EACH ROW EXECUTE FUNCTION update_company_clarity();


-- ══════════════════════════════════════════════════════════════
-- ROW LEVEL SECURITY (Supabase)
-- ══════════════════════════════════════════════════════════════

-- Activer RLS sur toutes les tables
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE events_raw ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE action_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics_history ENABLE ROW LEVEL SECURITY;

-- Policy : service_role a accès à tout (utilisé par l'API)
-- Les policies par utilisateur seront ajoutées quand on aura l'auth

CREATE POLICY "Service role full access" ON companies
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON agent_configs
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON events_raw
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON scan_history
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON decisions
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON actions
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON action_logs
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON runs
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON reports
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON signals
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON metrics_history
    FOR ALL USING (true) WITH CHECK (true);
