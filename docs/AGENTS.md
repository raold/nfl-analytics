# Agent Handbook

## 🎯 Overview

This NFL analytics platform uses a **three-agent coordination model** for efficient development and operations:

1. **DevOps Agent** - Infrastructure, database, deployment, monitoring
2. **ETL Agent** - Data ingestion, pipelines, validation, quality
3. **Research/Analytics Agent** - Modeling, features, backtesting, dissertation (R + Python + LaTeX)

## 📚 Documentation Structure

### Quick Start
- **[Quick Reference Guide](docs/agent_context/SUBAGENT_QUICK_REFERENCE.md)** ⚡ - Start here! Cheat sheet for common tasks

### Agent-Specific Guides
- **[DevOps Agent](docs/agent_context/SUBAGENT_DEVOPS.md)** 🔧 - Infrastructure & deployment
- **[ETL Agent](docs/agent_context/SUBAGENT_ETL.md)** 📊 - Data pipelines & quality
- **[Research/Analytics Agent](docs/agent_context/SUBAGENT_RESEARCH_ANALYTICS.md)** 🔬 - ML models & academic output

### Coordination
- **[Subagent Coordination](docs/agent_context/SUBAGENT_COORDINATION.md)** 🤝 - Handoffs, workflows, communication
- **[Main Agent Guide](docs/agent_context/AGENTS.md)** 📖 - Enterprise-level overview and technical details

## 🚀 Which Agent Are You?

### I handle infrastructure, databases, and deployments
→ You are the **DevOps Agent**. Read: `docs/agent_context/SUBAGENT_DEVOPS.md`

### I manage data pipelines, ingestion, and data quality
→ You are the **ETL Agent**. Read: `docs/agent_context/SUBAGENT_ETL.md`

### I build models, engineer features, and write papers
→ You are the **Research/Analytics Agent**. Read: `docs/agent_context/SUBAGENT_RESEARCH_ANALYTICS.md`

### I'm new and need to understand the whole system
→ Read the **Quick Reference** first, then the **Main Agent Guide**

## 🆘 Quick Help

**"Something is broken!"**
- System down? → DevOps Agent
- Pipeline failing? → ETL Agent  
- Model issues? → Research Agent
- Not sure? → Check [Quick Reference](docs/agent_context/SUBAGENT_QUICK_REFERENCE.md)

**"I need to coordinate with another agent"**
- Read: [Subagent Coordination](docs/agent_context/SUBAGENT_COORDINATION.md)
- Use handoff templates in that document

**"Where is X file or Y documentation?"**
- Check: [Quick Reference - File Ownership](docs/agent_context/SUBAGENT_QUICK_REFERENCE.md#-file-ownership-quick-map)

## 🎓 Learning Path

1. **Day 1**: Read Quick Reference + your agent-specific guide
2. **Day 2**: Read Coordination guide + Main Agent Guide  
3. **Week 1**: Shadow workflows, review handoff examples
4. **Ongoing**: Update docs as you learn, share insights

---

**Legacy Note**: The detailed enterprise guide lives in `docs/agent_context/AGENTS.md` for consolidated documentation.
