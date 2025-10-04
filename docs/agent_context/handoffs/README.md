# Handoff Templates

This directory contains templates for structured handoffs between agents. Using these templates ensures clear communication and complete information transfer.

## 📋 Available Templates

### 1. Data to Research Handoff
**File**: `template_etl_to_research.yaml`  
**Use When**: New or updated datasets are ready for modeling

### 2. Research to DevOps Deployment
**File**: `template_research_to_devops_deploy.yaml`  
**Use When**: Model is ready for production deployment

### 3. DevOps to ETL Infrastructure Change
**File**: `template_devops_to_etl_infra.yaml`  
**Use When**: Database or infrastructure changes affect data pipelines

### 4. Research to ETL Feature Request
**File**: `template_research_to_etl_feature.yaml`  
**Use When**: New features are needed for modeling

### 5. ETL to DevOps Resource Request
**File**: `template_etl_to_devops_resource.yaml`  
**Use When**: Pipeline needs more resources or infrastructure changes

### 6. Data Quality Issue Report
**File**: `template_data_quality_issue.yaml`  
**Use When**: Data quality problems are discovered

## 📝 How to Use

1. **Copy the appropriate template** to a new file with date and description:
   ```bash
   cp template_etl_to_research.yaml 2025-01-20_weekly_features.yaml
   ```

2. **Fill in all fields** with specific information

3. **Review** before sending to ensure completeness

4. **Send handoff** by:
   - Placing in `handoffs/active/`
   - Notifying recipient agent
   - Tracking in daily stand-up

5. **Archive** when complete:
   ```bash
   mv handoffs/active/2025-01-20_weekly_features.yaml handoffs/archive/2025-01/
   ```

## 🗂 Directory Structure

```
handoffs/
├── README.md                          # This file
├── templates/                         # Template files
│   ├── template_etl_to_research.yaml
│   ├── template_research_to_devops_deploy.yaml
│   ├── template_devops_to_etl_infra.yaml
│   ├── template_research_to_etl_feature.yaml
│   ├── template_etl_to_devops_resource.yaml
│   └── template_data_quality_issue.yaml
├── active/                           # Current handoffs in progress
│   └── .gitkeep
└── archive/                          # Completed handoffs by month
    ├── 2025-01/
    └── .gitkeep
```

## ✅ Best Practices

1. **Be Specific**: Provide exact file paths, table names, and metrics
2. **Include Context**: Explain why this handoff is happening now
3. **Set Expectations**: Define what action is required and by when
4. **Provide Impact**: Explain how this affects downstream work
5. **Acknowledge Receipt**: Receiving agent should confirm understanding
6. **Close the Loop**: Mark as complete when action is done

## 🔄 Handoff Lifecycle

```
Created → Sent → Acknowledged → In Progress → Completed → Archived
```

Each handoff should track its status and be updated as it progresses through these stages.

## 📊 Handoff Metrics

Track these metrics monthly:
- Average time to acknowledgment
- Average time to completion
- Number of clarifications needed (lower is better)
- Handoff completion rate

## 🚨 Escalation

If a handoff is not acknowledged within 24 hours or completed within agreed timeline:
1. Follow up directly with recipient agent
2. Raise in daily stand-up
3. If critical, escalate to human oversight

## 📞 Questions?

- Refer to [Subagent Coordination Guide](../SUBAGENT_COORDINATION.md)
- Raise in weekly planning session
- Update this README if process needs improvement
