# 🚀 How to Initialize a New Claude Session

When you open a new Claude window/session, the AI doesn't automatically know about your subagent system. Here's how to get oriented quickly.

---

## Method 1: Quick Prompt (Fastest - 30 seconds)

**Copy-paste this into your new Claude session**:

```
I'm working on an NFL analytics project with a three-agent system 
(DevOps, ETL, Research/Analytics). Please read:

1. .claude_context
2. docs/agent_context/SUBAGENT_QUICK_REFERENCE.md

I'm currently working as the [DevOps/ETL/Research] Agent.
```

Claude will read those files and understand the context.

---

## Method 2: Attach Files (Most Reliable - 1 minute)

1. **Start new Claude session**
2. **Click attachment button** (📎)
3. **Attach these files**:
   - `.claude_context`
   - `docs/agent_context/SUBAGENT_QUICK_REFERENCE.md`
   - Your agent handbook: `docs/agent_context/SUBAGENT_[YOUR_AGENT].md`
4. **Say**: "I'm the [Agent] agent. Review these docs and help me with my work."

---

## Method 3: Use This Starter Prompt Template

Save this as a snippet/template for reuse:

```markdown
# New Session Initialization

**Project**: NFL Analytics with Three-Agent System

**My Role Today**: [DevOps / ETL / Research] Agent

**Context Files**:
- .claude_context (project overview)
- docs/agent_context/SUBAGENT_QUICK_REFERENCE.md (cheat sheet)
- docs/agent_context/SUBAGENT_[MY_AGENT].md (my handbook)

**Current Task**: [Describe what you're working on]

**Recent Context**: [Any relevant recent work or decisions]

**What I Need Help With**: [Specific ask]

Please review the context files and help me with this task following 
the responsibilities and processes defined for my agent role.
```

---

## What Claude Needs to Know

For Claude to be effective, it needs:

1. ✅ **Which agent you are** - DevOps, ETL, or Research
2. ✅ **The agent system exists** - Point to documentation
3. ✅ **Current task** - What you're working on
4. ✅ **Recent context** - Any handoffs or recent work

---

## Example Session Starts

### DevOps Agent Starting a Session
```
Hi! I'm the DevOps Agent for an NFL analytics project.

Please read:
- .claude_context
- docs/agent_context/SUBAGENT_QUICK_REFERENCE.md
- docs/agent_context/SUBAGENT_DEVOPS.md

I need to create a database migration to add new columns for 
weather data. The ETL agent requested this in a handoff.
```

### ETL Agent Starting a Session
```
Hi! I'm the ETL Agent for an NFL analytics project.

Please read:
- .claude_context
- docs/agent_context/SUBAGENT_QUICK_REFERENCE.md
- docs/agent_context/SUBAGENT_ETL.md

I need to ingest Week 5 NFL data and validate the quality 
before handing off to Research.
```

### Research Agent Starting a Session
```
Hi! I'm the Research/Analytics Agent for an NFL analytics project.

Please read:
- .claude_context
- docs/agent_context/SUBAGENT_QUICK_REFERENCE.md
- docs/agent_context/SUBAGENT_RESEARCH_ANALYTICS.md

I need to train an XGBoost model and prepare results for my 
dissertation. Need help generating LaTeX tables from backtest results.
```

---

## What About `/init`?

`/init` is typically for **system initialization** (installing packages, starting services), NOT for loading documentation context.

Use `/init` if you need to:
- Set up Python environment
- Install dependencies
- Start Docker containers
- Run database migrations

But it won't help Claude understand your agent system.

---

## Best Practice Workflow

### Starting Your Work Day

1. **Open Claude Desktop**
2. **Use quick prompt** (Method 1 above)
3. **Declare your agent role**
4. **Describe your current task**
5. **Start working!**

### Switching Agent Roles

If you need to switch from DevOps to ETL (for example):

```
I'm now switching to ETL Agent mode. 
Please read docs/agent_context/SUBAGENT_ETL.md 
and focus on data pipeline responsibilities.
```

### Mid-Session Context Refresh

If Claude seems to forget the context:

```
Reminder: I'm the [Agent] agent. Please refer back to 
docs/agent_context/SUBAGENT_[AGENT].md for my responsibilities.
```

---

## Pro Tips

### 1. Create a Keyboard Shortcut
Save your initialization prompt in a text file and use a keyboard shortcut tool (like TextExpander on Mac) to paste it quickly.

### 2. Update `.claude_context` Regularly
Keep `.claude_context` current with:
- What you're working on
- Recent decisions
- Current blockers

### 3. Reference Specific Sections
If you need Claude to focus on something specific:

```
I'm the DevOps agent. Please read the "SOP-001: Database Migration" 
section from docs/agent_context/SUBAGENT_DEVOPS.md and help me 
execute that procedure.
```

### 4. Use Handoffs for Context
If you're responding to a handoff:

```
I'm the ETL agent. There's a handoff from Research in 
docs/agent_context/handoffs/active/2025-10-04_feature_request.yaml
Please read it and help me implement the requested feature.
```

---

## Troubleshooting

### "Claude doesn't seem to understand the agent system"
→ Explicitly attach the Quick Reference or your agent handbook

### "Claude is giving answers for the wrong agent"
→ Clearly state: "I'm the [Agent] agent, not [Other Agent]"

### "I need to switch contexts frequently"
→ Use the switching prompt template above

### "Documentation is too long to load every time"
→ Just load Quick Reference + your agent handbook, not everything

---

## Quick Reference for Session Initialization

**Minimum Viable Context**:
```
I'm the [Agent] agent.
Read: .claude_context and docs/agent_context/SUBAGENT_QUICK_REFERENCE.md
Task: [what you're doing]
```

**Full Context**:
```
I'm the [Agent] agent.
Read: 
- .claude_context
- docs/agent_context/SUBAGENT_QUICK_REFERENCE.md
- docs/agent_context/SUBAGENT_[MY_AGENT].md
Task: [what you're doing]
Recent: [recent context]
Need: [specific help]
```

---

## Remember

The documentation is **already in your repository**. You're just telling each new Claude session where to find it and which role you're playing.

Think of it like this:
- **The files** = Your operating manual (persistent)
- **The session initialization** = Handing the manual to a new assistant (per session)
- **Your agent declaration** = "I'm the mechanic, not the electrician" (role clarity)

---

**Save this file and refer to it every time you start a new Claude session!**
