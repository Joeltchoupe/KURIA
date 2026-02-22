-- ══════════════════════════════════════════════════════════════
-- KURIA — Données de test
-- ══════════════════════════════════════════════════════════════

-- Company de test
INSERT INTO companies (id, name, industry, size, growth_stage, connected_tools, primary_contact_name, primary_contact_email, clarity_score)
VALUES (
    'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    'Acme Corp',
    'B2B SaaS',
    'small',
    'scaling',
    '["hubspot", "quickbooks", "gmail"]'::JSONB,
    'Jean Dupont',
    'jean@acme.com',
    47
);

-- Config agents de test
INSERT INTO agent_configs (company_id, config_data, revenue_velocity_enabled, process_clarity_enabled, cash_predictability_enabled, acquisition_efficiency_enabled)
VALUES (
    'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    '{
        "revenue_velocity": {
            "enabled": true,
            "stagnation_threshold_days": 21,
            "zombie_threshold_days": 45,
            "hot_lead_threshold": 75
        },
        "process_clarity": {
            "enabled": true,
            "sop_min_occurrences": 5,
            "max_tasks_per_person_warning": 10,
            "hourly_rate_estimate": 50
        },
        "cash_predictability": {
            "enabled": true,
            "cash_threshold_yellow_months": 3.0,
            "cash_threshold_red_months": 1.5
        },
        "acquisition_efficiency": {
            "enabled": false
        }
    }'::JSONB,
    TRUE, TRUE, TRUE, FALSE
);

-- Scan initial de test
INSERT INTO scan_history (company_id, scan_number, scan_mode, clarity_score, clarity_details, top_frictions, estimated_annual_waste)
VALUES (
    'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    1,
    'initial',
    47,
    '{
        "machine_readability": 52,
        "structural_compatibility": 42,
        "data_quality": 60,
        "data_completeness": 35,
        "process_explicitness": 30,
        "tool_integration": 45
    }'::JSONB,
    '[
        {"title": "30% des deals sans montant", "impact_monthly": 5000, "difficulty": "low", "status": "new"},
        {"title": "Process non documentés", "impact_monthly": 8000, "difficulty": "medium", "status": "new"},
        {"title": "Données marketing absentes", "impact_monthly": 3000, "difficulty": "high", "status": "new"}
    ]'::JSONB,
    192000
);

-- Quelques events de test
INSERT INTO events_raw (company_id, event_type, source, actor, payload) VALUES
('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'deal.created', 'hubspot', 'jean@acme.com', '{"deal_id": "d001", "name": "BigCorp", "amount": 50000, "stage": "qualification"}'::JSONB),
('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'deal.stage_changed', 'hubspot', 'jean@acme.com', '{"deal_id": "d001", "old_stage": "qualification", "new_stage": "proposal_sent"}'::JSONB),
('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'invoice.created', 'quickbooks', 'system', '{"invoice_id": "INV-001", "amount": 15000, "client": "OldClient SA"}'::JSONB),
('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'task.created', 'hubspot', 'marie@acme.com', '{"task_id": "t001", "title": "Préparer onboarding", "assignee": "marie@acme.com"}'::JSONB);
