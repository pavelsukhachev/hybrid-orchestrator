# The Hybrid Orchestrator: A Framework for Coordinating Human-AI Teams

**Pavel Sukhachev**
Electromania LLC
pavel@electromania.llc
ORCID: 0009-0006-4546-9963

---

## Abstract

Long-running AI agents face a fundamental constraint: context windows limit what they can remember in a single session. Recent work addresses this for software development through external task trackers. But real enterprises need more than coding agents. They need hybrid teams—humans and AI working together across voice, text, and web.

This paper presents the Hybrid Orchestrator, a framework for coordinating human-AI teams. The framework has three layers: an orchestrator that monitors progress and detects blockers, workers (both human and AI) that execute tasks, and communication channels that connect them. We describe four design patterns that make this work: (1) session state externalization for cross-session continuity, (2) multi-channel communication routing, (3) real-time activity monitoring with configurable triggers, and (4) human escalation pathways.

We do not claim these patterns are novel in isolation. Monitoring, notifications, and escalation are well-established. Our contribution is documenting how these patterns combine into a working framework for hybrid teams, with implementation details and a reference implementation under Apache 2.0.

**Keywords**: multi-agent systems, human-AI collaboration, hybrid orchestration, enterprise automation, design patterns

---

## 1. Introduction

### 1.1 The Context Window Problem

Large language models operate within fixed context windows. Claude supports 200,000 tokens. GPT-4 supports 128,000 tokens. When a task requires more context than available, the agent must start a new session, losing accumulated state.

This creates practical problems for long-running tasks:

1. **Repeated exploration**. New sessions re-discover information the previous session already found.
2. **Lost decisions**. Choices made in earlier sessions are not available later.
3. **Broken continuity**. Multi-step workflows stall when context resets.

### 1.2 Beyond Coding Agents

Recent work addresses the context window problem for software development. Anthropic's engineering blog recommends "externalizing state to persistent storage" (Anthropic, 2025). The Linear Agent Harness (Medin, 2025) implements this—agents use Linear's issue tracker as external memory.

This works well for coding. But enterprises need more:

- **Hybrid teams**. Real work involves humans and AI agents together. A coding agent works alone. An enterprise workflow has managers, specialists, and AI each handling different parts.
- **Multiple channels**. Coding agents communicate through task comments. Enterprise teams use voice, email, SMS, Slack, and dashboards.
- **Domain adaptation**. A coding agent framework assumes pull requests and test suites. Insurance needs forms and compliance. Healthcare needs patient records and HIPAA. Each domain has different workflows.

No existing framework addresses all three.

### 1.3 Our Contribution

We present the Hybrid Orchestrator, a framework that coordinates human-AI teams across multiple communication channels with pluggable domain adapters.

Our contributions:

1. **Framework Architecture**. A three-layer system: orchestrator, workers, and channels.
2. **Four Design Patterns**. Documented patterns with implementation details.
3. **Domain Adapter Pattern**. A pluggable architecture for industry-specific customization.
4. **Reference Implementation**. Open-source code under Apache 2.0.

We explicitly acknowledge:
- These patterns are not individually novel
- We have not conducted controlled experiments
- Generalization to diverse domains requires further validation

### 1.4 Paper Organization

Section 2 reviews related work. Section 3 presents the framework architecture. Section 4 details the four design patterns. Section 5 describes the domain adapter system. Section 6 covers implementation. Section 7 discusses limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 Long-Running Agent Frameworks

**The Linear Agent Harness** (Medin, 2025) uses Linear's issue tracker as external memory for coding agents. Key design decisions:

- Two-agent pattern: initializer creates issues, coding agent implements them
- Status transitions provide workflow structure
- Comments preserve context between sessions

The demo video shows the system building a Claude.ai clone over 24 hours, completing approximately 54% of 200 tasks. This demonstrates both viability and limitations—agents sometimes loop, hallucinate, or require intervention.

Our work extends this approach beyond coding. We add voice communication and human worker coordination. However, we inherit similar limitations: agents require monitoring and occasional intervention.

### 2.2 Multi-Agent Frameworks

**LangGraph** uses directed graphs to define agent workflows. It supports human-in-the-loop patterns through interruptible nodes.

**AutoGen** (Microsoft) enables conversational multi-agent systems. It explicitly supports human agents alongside AI agents. The `UserProxyAgent` class provides human integration. Our contribution is not "adding humans to multi-agent systems"—AutoGen already does this.

**CrewAI** assigns roles to agents in a crew structure. It focuses on AI-to-AI delegation rather than human-AI coordination.

These frameworks are more general than ours. We solve a narrower problem (coordinating hybrid teams across multiple channels) with a more specific solution.

### 2.3 Voice AI Platforms

**VAPI** provides enterprise voice AI infrastructure with telephony integration. **OpenAI Realtime API** offers WebSocket-based voice interaction. These platforms provide voice capabilities but not orchestration.

Our contribution is not voice capability—these platforms provide that—but the integration of voice with session tracking, activity monitoring, and human escalation.

### 2.4 Enterprise Workflow Tools

Traditional workflow tools (ServiceNow, Salesforce Flow) support human tasks, approvals, and notifications. These systems have solved human-AI coordination for decades. Our contribution is applying similar patterns in an LLM-agent context, not inventing new patterns.

### 2.5 What We Actually Contribute

| Pattern | Prior Art | Our Contribution |
|---------|-----------|------------------|
| External state storage | Linear Agent Harness, databases | Generalize beyond coding |
| Human-AI coordination | AutoGen, workflow tools | Integrate with voice + multiple channels |
| Multi-channel notifications | Twilio, PagerDuty | Combine in agent orchestration context |
| Activity monitoring | APM tools, dashboards | Apply to hybrid team workflows |

Our contribution is the combination and documentation, not the individual patterns.

---

## 3. Framework Architecture

### 3.1 Three-Layer Architecture

The Hybrid Orchestrator has three layers:

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Monitor    │  │   Blocker    │  │   Channel    │  │
│  │   Service    │  │   Detector   │  │   Selector   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
├─────────┼─────────────────┼─────────────────┼───────────┤
│         │          WORKER LAYER              │           │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐  │
│  │  AI Worker   │  │  AI Worker   │  │ Human Worker │  │
│  │  (Agent 1)   │  │  (Agent 2)   │  │  (Employee)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
├─────────┼─────────────────┼─────────────────┼───────────┤
│         │         CHANNEL LAYER              │           │
│  ┌──────┴──┐  ┌───┴───┐  ┌──┴───┐  ┌───────┴───────┐  │
│  │  Voice  │  │  SMS  │  │ Email│  │   Dashboard   │  │
│  └─────────┘  └───────┘  └──────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Orchestrator Layer**: Monitors all workers. Detects blockers. Selects the right communication channel for each intervention.

**Worker Layer**: Executes tasks. Workers can be AI agents or human employees. Both report status to the same task tracker.

**Channel Layer**: Delivers messages. Each channel has different latency, richness, and appropriate use cases.

### 3.2 Component Roles

| Component | Role | Example |
|-----------|------|---------|
| **Monitor Service** | Watches task progress, detects stalls | Polls task tracker every 30 seconds |
| **Blocker Detector** | Identifies what is blocking progress | "Task stalled 5 minutes, same screen" |
| **Channel Selector** | Picks the best channel for the situation | Voice for urgent, SMS for links, email for details |
| **AI Worker** | Executes tasks autonomously | Fills forms, writes code, researches |
| **Human Worker** | Handles exceptions the AI cannot | Reviews edge cases, approves decisions |
| **Task Tracker** | Source of truth for all work | Database, Linear, JIRA |

### 3.3 Data Flow

A typical interaction:

1. **Task created** in tracker (by user, API, or another agent)
2. **AI worker picks up** the task and begins work
3. **Monitor service** watches for progress
4. **AI worker reports** status updates to tracker
5. **Blocker detected** (e.g., no progress for 5 minutes)
6. **Channel selector** picks voice (urgent) or SMS (link needed)
7. **Intervention delivered** to the right person via the right channel
8. **Human worker** takes over if AI cannot resolve
9. **Task completed** and marked done in tracker

### 3.4 Blocker Detection Logic

```python
def detect_blocker(task: Task) -> Optional[Blocker]:
    # Time-based: task stuck too long
    if task.time_in_status > threshold_by_type[task.type]:
        return Blocker(type="stalled", severity="medium")

    # Failure-based: repeated errors
    if task.consecutive_failures > 3:
        return Blocker(type="repeated_failure", severity="high")

    # Dependency-based: waiting on something
    if task.blocked_by and task.blocked_by.status != "done":
        return Blocker(type="dependency", severity="low")

    # Inactivity-based: user went silent
    if task.last_user_activity and \
       (now() - task.last_user_activity).seconds > 300:
        return Blocker(type="user_inactive", severity="medium")

    return None
```

### 3.5 Channel Selection Logic

```python
def select_channel(blocker: Blocker, worker: Worker) -> str:
    if blocker.severity == "high" and worker.is_available:
        return "voice"     # Immediate, high-bandwidth

    if blocker.type == "link_needed":
        return "sms"       # Links work best via text

    if blocker.type == "detailed_instructions":
        return "email"     # Long-form content

    if worker.preferred_channel:
        return worker.preferred_channel  # Respect preferences

    if blocker.severity == "low":
        return "dashboard"  # Don't interrupt

    return "sms"           # Default for medium severity
```

---

## 4. Design Patterns

We present four patterns that form the core of the framework.

### 4.1 Pattern 1: Session State Externalization

**Problem**: Agent sessions are ephemeral. Context windows fill up. Sessions timeout. How do you maintain state across session boundaries?

**Solution**: Store all session state in a database. The agent queries current state at session start, updates state throughout, writes summary before session ends.

**Implementation**:

```sql
-- Session table stores all state external to the agent
CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type   VARCHAR(50) NOT NULL,     -- Domain identifier
    external_ref    VARCHAR(255) UNIQUE,      -- External system reference
    worker_id       VARCHAR(100),             -- Current assigned worker
    status          VARCHAR(20) DEFAULT 'ACTIVE',
    context         JSONB NOT NULL DEFAULT '{}',  -- Domain-specific state
    pending_actions JSONB,                    -- Queued commands
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    expires_at      TIMESTAMP
);

-- Activity log enables session reconstruction
CREATE TABLE activities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    actor_type      VARCHAR(20) NOT NULL,     -- 'ai', 'human', 'system'
    action          VARCHAR(100) NOT NULL,
    data            JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_activities_session ON activities(session_id);
CREATE INDEX idx_activities_created ON activities(created_at);
```

**Key Design Decisions**:

1. **Generic context field**. Domain-specific state goes in `context` (JSONB). No need to change the schema for each domain.

2. **Actor type tracking**. Every activity records who did it—AI, human, or system. This enables audit trails and debugging.

3. **Activity log is append-only**. Never update; always insert. Enables replay and debugging.

4. **Expiration built in**. Sessions auto-expire. Cleanup is automatic.

**Tradeoffs**:

- Adds database dependency (latency, failure modes)
- Requires careful schema design upfront
- Storage grows with activity volume
- JSONB queries are slower than typed columns

### 4.2 Pattern 2: Multi-Channel Communication Hub

**Problem**: Different situations need different communication channels. Voice for urgent guidance. SMS for links. Dashboard for monitoring. How do you coordinate?

**Solution**: Create a central hub that routes messages to appropriate channels based on context.

**Implementation**:

```python
class ChannelHub:
    def __init__(self):
        self.channels = {
            'voice': VoiceChannel(),
            'sms': SMSChannel(),
            'email': EmailChannel(),
            'dashboard': DashboardChannel(),
            'slack': SlackChannel(),
            'task_tracker': TaskTrackerChannel()
        }

    async def send(self, message: str, context: MessageContext):
        channel_name = select_channel(context.blocker, context.recipient)
        channel = self.channels[channel_name]

        try:
            result = await channel.send(message, context.recipient)
            await self.log_delivery(channel_name, message, result)
            return result
        except ChannelError:
            # Fallback to next-best channel
            fallback = self.get_fallback(channel_name)
            return await self.channels[fallback].send(message, context.recipient)

    def get_fallback(self, failed_channel: str) -> str:
        fallbacks = {
            'voice': 'sms',
            'sms': 'email',
            'email': 'dashboard',
            'slack': 'email'
        }
        return fallbacks.get(failed_channel, 'dashboard')
```

**Channel Comparison**:

| Channel | Latency | Richness | Best For | Fallback |
|---------|---------|----------|----------|----------|
| Voice | Real-time | High | Urgent guidance, complex explanations | SMS |
| SMS | Seconds | Low | Links, short updates, confirmations | Email |
| Email | Minutes | High | Detailed instructions, documents | Dashboard |
| Dashboard | Real-time | Medium | Monitoring, status overview | Email |
| Slack | Seconds | Medium | Team coordination, quick questions | Email |

**Lesson Learned**: Voice AI platforms are brittle to HTTP errors. Early versions returned 500 on errors; this caused voice sessions to disconnect entirely. Always return 200 with error details in the payload:

```python
@app.post('/webhooks/voice')
async def handle_voice_webhook(request):
    try:
        result = await process_tool_call(request)
        return JSONResponse(status_code=200, content=result)
    except Exception:
        # CRITICAL: Always return 200 to voice platforms
        return JSONResponse(status_code=200, content={
            "success": False,
            "error": "Temporary issue. Please try again."
        })
```

### 4.3 Pattern 3: Activity Monitoring with Triggers

**Problem**: How do you detect when a user or worker is stuck, confused, or needs help?

**Solution**: Track the activity stream. Apply rules to detect patterns. Trigger interventions when patterns match.

**Implementation**:

```python
class ActivityMonitor:
    def __init__(self, rules: list[TriggerRule]):
        self.rules = rules
        self.trigger_counts = {}  # Track per-session trigger counts

    async def check(self, session: Session) -> Optional[Intervention]:
        activities = await get_recent_activities(session.id)

        for rule in self.rules:
            if self._should_trigger(rule, session, activities):
                return Intervention(
                    type=rule.action_type,
                    channel=rule.channel,
                    message=rule.message.format(**session.context),
                    priority=rule.priority
                )
        return None

    def _should_trigger(self, rule, session, activities) -> bool:
        # Check max triggers per session
        key = f"{session.id}:{rule.name}"
        if self.trigger_counts.get(key, 0) >= rule.max_triggers:
            return False

        # Evaluate condition
        if rule.condition_type == "no_activity":
            return self._check_inactivity(activities, rule.duration_seconds)
        elif rule.condition_type == "repeated_error":
            return self._check_repeated_errors(activities, rule.error_count)
        elif rule.condition_type == "no_progress":
            return self._check_no_progress(activities, rule.duration_seconds)

        return False
```

**Trigger Rules** (configurable per domain):

```yaml
triggers:
  - name: gentle_nudge
    condition_type: no_activity
    duration_seconds: 120
    action_type: voice_prompt
    channel: voice
    message: "Still there? Need any help?"
    max_triggers: 2
    priority: low

  - name: repeated_error
    condition_type: repeated_error
    error_count: 3
    within_seconds: 60
    action_type: guidance
    channel: voice
    message: "Looks like that field is tricky. Let me help."
    max_triggers: 1
    priority: medium

  - name: escalate_to_human
    condition_type: no_progress
    duration_seconds: 300
    action_type: dashboard_alert
    channel: dashboard
    message: "Session {id} stuck for 5+ minutes. Needs human review."
    max_triggers: 1
    priority: high

  - name: followup_after_abandon
    condition_type: no_activity
    duration_seconds: 3600
    action_type: sms
    channel: sms
    message: "Hi {name}! Ready to continue where you left off? {resume_link}"
    max_triggers: 1
    priority: low
```

**Important**: We have not validated these specific thresholds experimentally. The values are based on informal observation, not controlled studies.

### 4.4 Pattern 4: Human Escalation Pathways

**Problem**: AI agents cannot handle everything. How do you smoothly escalate to humans?

**Solution**: Define escalation triggers. Route to human queue. Provide full context. Enable seamless takeover.

**Implementation**:

```python
class EscalationManager:
    async def escalate(self, session: Session, reason: str):
        # Collect full context for the human
        context = await self.build_context(session)

        # Create dashboard alert
        alert = DashboardAlert(
            session_id=session.id,
            reason=reason,
            context=context,
            priority="high",
            created_at=now()
        )
        await self.dashboard.push_alert(alert)

        # Notify available human workers
        humans = await self.get_available_humans(session.workflow_type)
        for human in humans:
            await self.channel_hub.send(
                f"Session {session.id} needs attention: {reason}",
                MessageContext(recipient=human, urgency="high")
            )

    async def build_context(self, session: Session) -> dict:
        """Give humans everything they need. Never make them ask the user to repeat."""
        activities = await get_all_activities(session.id)
        return {
            "session": session.to_dict(),
            "activity_timeline": [a.to_dict() for a in activities],
            "current_state": session.context,
            "time_elapsed": (now() - session.created_at).total_seconds(),
            "ai_actions_taken": [a for a in activities if a.actor_type == "ai"],
            "escalation_reason": session.last_blocker
        }
```

**Human Takeover Flow**:

1. Dashboard highlights sessions needing attention
2. Human sees full context (all activities, current state, what AI tried)
3. Human can: send a message, make a call, or take over the session directly
4. If voice takeover: call transfers to human agent queue
5. Human never asks the user to repeat information—the dashboard shows everything

**Design Principle**: The human should know more about the session than the user does. Complete context transfer is not optional.

---

## 5. Domain Adapters

### 5.1 Adapter Architecture

Different industries have different workflows. The Hybrid Orchestrator uses a domain adapter pattern to handle this.

```python
class DomainAdapter:
    """Base class for domain-specific customization."""

    def get_trigger_rules(self) -> list[TriggerRule]:
        """Return domain-specific trigger rules."""
        raise NotImplementedError

    def get_worker_roles(self) -> list[WorkerRole]:
        """Define the types of workers needed."""
        raise NotImplementedError

    def get_channel_preferences(self) -> dict:
        """Map situations to preferred channels."""
        raise NotImplementedError

    def parse_activity(self, raw_data: dict) -> Activity:
        """Convert domain-specific events to standard activities."""
        raise NotImplementedError

    def detect_domain_blockers(self, session: Session) -> Optional[Blocker]:
        """Domain-specific blocker detection beyond standard rules."""
        return None
```

### 5.2 Example: Financial Services Adapter

Financial services workflows involve form completion, document verification, and compliance checks.

```python
class FinancialServicesAdapter(DomainAdapter):
    def get_trigger_rules(self):
        return [
            TriggerRule(
                name="form_field_stuck",
                condition_type="same_field_changed",
                times=3, within_seconds=60,
                action_type="highlight_and_guide",
                channel="voice"
            ),
            TriggerRule(
                name="compliance_review",
                condition_type="reached_stage",
                stage="submission",
                action_type="human_review",
                channel="dashboard"
            )
        ]

    def get_worker_roles(self):
        return [
            WorkerRole("voice_guide", type="ai",
                        description="Guides users through forms via phone"),
            WorkerRole("compliance_reviewer", type="human",
                        description="Reviews submissions for compliance"),
            WorkerRole("supervisor", type="human",
                        description="Monitors all sessions, handles escalations")
        ]

    def get_channel_preferences(self):
        return {
            "form_guidance": "voice",
            "send_link": "sms",
            "compliance_alert": "dashboard",
            "followup": "sms"
        }
```

### 5.3 Example: Software Development Adapter

Extends the Linear Agent Harness pattern with human code reviewers.

```python
class SoftwareDevelopmentAdapter(DomainAdapter):
    def get_trigger_rules(self):
        return [
            TriggerRule(
                name="tests_failing",
                condition_type="repeated_error",
                error_count=3,
                action_type="escalate",
                channel="slack"
            ),
            TriggerRule(
                name="pr_stale",
                condition_type="no_activity",
                duration_seconds=86400,  # 24 hours
                action_type="nudge",
                channel="slack"
            )
        ]

    def get_worker_roles(self):
        return [
            WorkerRole("coding_agent", type="ai",
                        description="Writes code, runs tests"),
            WorkerRole("code_reviewer", type="human",
                        description="Reviews pull requests"),
            WorkerRole("tech_lead", type="human",
                        description="Resolves architectural blockers")
        ]
```

### 5.4 Example: Customer Support Adapter

```python
class CustomerSupportAdapter(DomainAdapter):
    def get_trigger_rules(self):
        return [
            TriggerRule(
                name="sentiment_drop",
                condition_type="sentiment_below",
                threshold=0.3,
                action_type="escalate",
                channel="dashboard"
            ),
            TriggerRule(
                name="topic_out_of_scope",
                condition_type="classification",
                label="out_of_scope",
                action_type="transfer_to_human",
                channel="voice"
            )
        ]

    def get_worker_roles(self):
        return [
            WorkerRole("frontline_ai", type="ai",
                        description="Handles common queries"),
            WorkerRole("specialist", type="human",
                        description="Handles complex issues"),
            WorkerRole("supervisor", type="human",
                        description="Monitors quality, handles complaints")
        ]
```

### 5.5 Writing Your Own Adapter

To add a new domain:

1. Subclass `DomainAdapter`
2. Define trigger rules for your workflow
3. Define worker roles (AI and human)
4. Set channel preferences
5. Implement `parse_activity` for your event format
6. Optionally add domain-specific blocker detection

The framework handles monitoring, routing, and escalation. Your adapter handles domain logic.

---

## 6. Implementation

### 6.1 Reference Implementation

We provide a reference implementation at `github.com/pavelsukhachev/hybrid-orchestrator` (Apache 2.0).

**What is Included**:
- Session state management (PostgreSQL)
- Activity tracking and change detection
- Trigger rule engine with configurable rules
- Channel hub with fallback logic
- Dashboard components
- Three example domain adapters
- 97 passing tests

**Repository Structure**:

```
hybrid-orchestrator/
├── src/
│   ├── orchestrator/       # Monitor, blocker detector, channel selector
│   ├── workers/            # AI and human worker base classes
│   ├── channels/           # Voice, SMS, email, dashboard, Slack
│   ├── adapters/           # Domain adapter base + examples
│   └── storage/            # Session and activity persistence
├── tests/                  # 97 tests
├── examples/               # Working examples with mock services
├── paper/                  # This paper
├── LICENSE                 # Apache 2.0
└── CITATION.cff            # Citation metadata
```

### 6.2 Technology Stack

```
Core Framework:
├── Python 3.11+
├── PostgreSQL 15+
├── asyncio for concurrent monitoring

Integrations (bring your own):
├── Voice: VAPI, Twilio, or similar
├── SMS: Twilio, MessageBird
├── Email: SendGrid, SES
├── Chat: Slack, Teams
├── Task Tracker: Linear, JIRA, or custom

Deployment:
├── Docker containers
├── Any cloud provider
└── ~$50-100/month infrastructure cost
```

### 6.3 Getting Started

```python
from hybrid_orchestrator import Orchestrator
from hybrid_orchestrator.adapters import FinancialServicesAdapter
from hybrid_orchestrator.channels import SMSChannel, DashboardChannel

# 1. Choose your domain
adapter = FinancialServicesAdapter()

# 2. Set up channels
channels = {
    'sms': SMSChannel(api_key="..."),
    'dashboard': DashboardChannel(ws_url="..."),
}

# 3. Create orchestrator
orchestrator = Orchestrator(
    adapter=adapter,
    channels=channels,
    db_url="postgresql://...",
    poll_interval=30  # seconds
)

# 4. Start monitoring
await orchestrator.start()
```

### 6.4 Security Considerations

For enterprise deployment:

**Authentication & Authorization**:
- API keys for all external integrations
- Role-based access to dashboard
- Session tokens with expiration
- Worker identity verification

**Data Protection**:
- Encrypt PII at rest (database encryption)
- TLS for all network communication
- Audit logging for compliance
- Data retention policies

**Voice AI Specific**:
- Do not log full transcripts (PII exposure)
- Mask phone numbers in logs
- Review voice recording consent policies

**Compliance** (domain-dependent):
- HIPAA may apply (healthcare)
- SOC2 recommended for enterprise
- PCI DSS for payment data
- Domain-specific regulations

---

## 7. Limitations

### 7.1 Evaluation Limitations

- **No controlled experiment**. We cannot claim hybrid outperforms alternatives.
- **No benchmark results yet**. We have published an evaluation benchmark (Sukhachev, 2026) but have not yet run comparative evaluations.
- **Framework is new**. Limited production validation across domains.

### 7.2 Technical Limitations

- **Voice latency**. Voice AI platforms add 500-1500ms per turn. Users notice.
- **Model costs**. Frontier models (GPT-4, Claude) are expensive at scale.
- **Brittleness**. Voice AI fails on background noise, accents, and interruptions.
- **Single orchestrator**. Current design uses one monitor process. Scaling requires distributed coordination.

### 7.3 Generalization Limitations

- **Domain adapters are untested at scale**. The adapter pattern is designed for flexibility but has not been validated across many domains.
- **English-only**. We have not tested multilingual scenarios.
- **Trigger thresholds are heuristic**. Optimal values likely differ by domain, user population, and workflow.

### 7.4 What We Do Not Know

- Whether the orchestration layer adds value over simpler approaches
- Optimal trigger thresholds for different domains
- How patterns scale beyond hundreds of concurrent sessions
- Long-term user satisfaction with hybrid workflows

---

## 8. Conclusion

We presented the Hybrid Orchestrator, a framework for coordinating human-AI teams. The framework has three layers (orchestrator, workers, channels) and four design patterns:

1. **Session State Externalization**: Store all state in a database for cross-session continuity.
2. **Multi-Channel Communication Hub**: Route messages to appropriate channels based on context and urgency.
3. **Activity Monitoring with Triggers**: Detect patterns in the activity stream and trigger interventions.
4. **Human Escalation Pathways**: Enable smooth handoff to human workers with complete context.

The domain adapter pattern makes the framework pluggable across industries. We showed examples for financial services, software development, and customer support.

These patterns are not novel individually. Our contribution is documenting their combination into a cohesive framework and providing a working implementation.

We release our reference implementation under Apache 2.0 and invite the community to validate, extend, and improve upon these patterns.

**Code**: github.com/pavelsukhachev/hybrid-orchestrator
**Benchmark**: huggingface.co/datasets/pashas/insurance-ai-reliability-benchmark

**Acknowledgments**: We thank Cole Medin for the Linear Agent Harness, which inspired this work. We thank the open-source community for feedback on earlier drafts.

**Conflict of Interest**: The author is founder of Electromania LLC, which develops enterprise AI systems.

---

## References

1. Anthropic. (2025). Effective Harnesses for Long-Running Agents. Anthropic Engineering Blog.

2. Medin, C. (2025). Linear Coding Agent Harness. GitHub. https://github.com/coleam00/Linear-Coding-Agent-Harness

3. Dellermann, D., Ebel, P., Söllner, M., & Leimeister, J. M. (2019). Hybrid Intelligence. Business & Information Systems Engineering, 61(5), 637-643.

4. Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. Microsoft Research.

5. VAPI. (2025). Enterprise Voice AI Platform. https://vapi.ai

6. LangGraph Documentation. (2024). https://github.com/langchain-ai/langgraph

7. Sukhachev, P. (2026). Insurance AI Reliability Benchmark. Hugging Face Hub. https://huggingface.co/datasets/pashas/insurance-ai-reliability-benchmark

---

## Appendix A: Session Schema

```sql
-- Full schema for the Hybrid Orchestrator

CREATE TYPE session_status AS ENUM ('ACTIVE', 'PAUSED', 'COMPLETED', 'CANCELLED', 'EXPIRED');
CREATE TYPE actor_type AS ENUM ('ai', 'human', 'system');

CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type   VARCHAR(50) NOT NULL,
    external_ref    VARCHAR(255) UNIQUE,
    worker_id       VARCHAR(100),
    worker_type     actor_type,
    status          session_status DEFAULT 'ACTIVE',
    context         JSONB NOT NULL DEFAULT '{}',
    pending_actions JSONB,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    expires_at      TIMESTAMP NOT NULL
);

CREATE TABLE activities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    actor           actor_type NOT NULL,
    action          VARCHAR(100) NOT NULL,
    data            JSONB NOT NULL DEFAULT '{}',
    channel         VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE trigger_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    rule_name       VARCHAR(100) NOT NULL,
    action_taken    VARCHAR(100) NOT NULL,
    channel_used    VARCHAR(50),
    result          VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);
CREATE INDEX idx_sessions_workflow ON sessions(workflow_type);
CREATE INDEX idx_activities_session ON activities(session_id);
CREATE INDEX idx_activities_created ON activities(created_at);
CREATE INDEX idx_trigger_log_session ON trigger_log(session_id);
```

## Appendix B: Trigger Rule Configuration

```yaml
# Example trigger configuration file

defaults:
  poll_interval_seconds: 30
  max_triggers_per_session: 5

triggers:
  # Gentle nudge for inactive users
  - name: gentle_nudge
    condition_type: no_activity
    duration_seconds: 90
    action_type: voice_prompt
    channel: voice
    message: "Still there? Let me know if you need help."
    max_triggers: 2
    priority: low

  # Guidance for repeated errors
  - name: error_guidance
    condition_type: repeated_error
    error_count: 2
    within_seconds: 60
    action_type: guidance
    channel: voice
    message: "That field can be tricky. Let me walk you through it."
    max_triggers: 1
    priority: medium

  # Human escalation for stuck sessions
  - name: escalate_stuck
    condition_type: no_progress
    duration_seconds: 300
    action_type: dashboard_alert
    channel: dashboard
    message: "Session {id} needs human review. Stuck for 5+ minutes."
    max_triggers: 1
    priority: high

  # Abandonment recovery via SMS
  - name: followup_sms
    condition_type: no_activity
    duration_seconds: 3600
    status_required: ACTIVE
    action_type: sms
    channel: sms
    message: "Hi {name}! Ready to continue? Click here: {resume_link}"
    max_triggers: 1
    priority: low
```

## Appendix C: Channel Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DeliveryResult:
    success: bool
    channel: str
    message_id: str = None
    error: str = None

class Channel(ABC):
    """Base interface for all communication channels."""

    @abstractmethod
    async def send(self, message: str, recipient: Worker) -> DeliveryResult:
        """Send a message to a worker."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this channel is currently operational."""
        pass

class VoiceChannel(Channel):
    async def send(self, message: str, recipient: Worker) -> DeliveryResult:
        # Initiate voice call or inject message into active call
        ...

class SMSChannel(Channel):
    async def send(self, message: str, recipient: Worker) -> DeliveryResult:
        # Send SMS via Twilio/MessageBird
        ...

class DashboardChannel(Channel):
    async def send(self, message: str, recipient: Worker) -> DeliveryResult:
        # Push alert to web dashboard via WebSocket
        ...

class SlackChannel(Channel):
    async def send(self, message: str, recipient: Worker) -> DeliveryResult:
        # Post message to Slack channel or DM
        ...

class EmailChannel(Channel):
    async def send(self, message: str, recipient: Worker) -> DeliveryResult:
        # Send email via SendGrid/SES
        ...
```
