# Hybrid Human-AI Orchestration

Design patterns for coordinating human workers and AI agents, with a working Python implementation.

[![Tests](https://img.shields.io/badge/tests-97%20passing-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

**Live demo:** [https://pavelsukhachev.github.io/hybrid-orchestrator/](https://pavelsukhachev.github.io/hybrid-orchestrator/)

## What This Is

A **reference implementation** of design patterns for hybrid human-AI systems:

1. **Session State Externalization** - Store agent state in SQLite for cross-session continuity
2. **Multi-Channel Communication** - Route messages to appropriate channels based on urgency
3. **Activity Monitoring with Triggers** - Detect user behavior patterns and trigger interventions
4. **Human Escalation Pathways** - Enable smooth handoff from AI to human agents

These patterns are extracted from a production voice AI system for insurance applications.

## What This Is NOT

- **Not a novel framework** - These are documented patterns, not new inventions
- **Not experimentally validated** - We have no controlled studies comparing to alternatives
- **Not production-ready** - This is a reference implementation for learning

## Quick Start

```bash
# Clone and install
git clone https://github.com/pavelsukhachev/hybrid-orchestrator.git
cd hybrid-orchestrator
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run demo
python examples/demo.py

# Run AI agent demo (uses mock agent, or set ANTHROPIC_API_KEY for real Claude)
python examples/agent_demo.py
```

## Usage

```python
from core import (
    Orchestrator,
    Trigger,
    TriggerCondition,
    TriggerAction,
    ConditionType,
    ActionType,
    ConsoleChannel,
    Recipient,
)

# Create orchestrator
orchestrator = Orchestrator(db_path="sessions.db")

# Register a channel
orchestrator.channels.register(ConsoleChannel())

# Add triggers
orchestrator.add_trigger(Trigger(
    name="inactivity_warning",
    condition=TriggerCondition(
        type=ConditionType.NO_ACTIVITY,
        params={"duration_seconds": 120},
    ),
    action=TriggerAction(
        type=ActionType.DASHBOARD_ALERT,
        params={"message": "User may need help"},
    ),
    max_fires_per_session=2,
    cooldown_seconds=60,
))

# Create a session
recipient = Recipient(id="user_123", name="John", email="john@example.com")
session = orchestrator.create_session(
    external_id="call_abc",
    metadata={"form_type": "application"},
    recipient=recipient,
)

# Record activities
orchestrator.record_activity(session.token, "field_change", {"field_id": "name"})
orchestrator.record_activity(session.token, "field_change", {"field_id": "email"})

# Check triggers (call periodically)
import asyncio
results = asyncio.run(orchestrator.check_triggers())
for r in results:
    if r.fired:
        print(f"Trigger {r.trigger_name} fired: {r.reason}")

# Complete session
orchestrator.complete(session.token)
```

## With AI Agent

```python
from core.agents.claude import ClaudeAgent, MockClaudeAgent

# Use real Claude (requires ANTHROPIC_API_KEY)
agent = ClaudeAgent()

# Or mock for testing
agent = MockClaudeAgent()

# Analyze session
response = await agent.analyze(
    session_summary="User filling insurance form, step 3 of 5",
    recent_activities=[
        {"type": "field_change", "data": {"field_id": "ssn"}},
        {"type": "field_change", "data": {"field_id": "ssn"}},
        {"type": "field_change", "data": {"field_id": "ssn"}},
    ],
    context={"form_type": "insurance_application"},
)

if response.action == ActionType.PROMPT_USER:
    print(f"AI suggests: {response.message}")
elif response.action == ActionType.ESCALATE:
    print(f"Escalating to human: {response.escalation_reason}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Python)                 │
│   SessionStore │ TriggerEngine │ ChannelHub │ Agent     │
└─────────────────────────────────────────────────────────┘
         │               │              │           │
         ▼               ▼              ▼           ▼
    ┌─────────┐    ┌──────────┐   ┌─────────┐  ┌─────────┐
    │ SQLite  │    │ Triggers │   │Channels │  │ Claude  │
    │Sessions │    │ Rules    │   │ Hub     │  │  API    │
    └─────────┘    └──────────┘   └─────────┘  └─────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
         ┌─────────┐             ┌──────────┐             ┌─────────┐
         │ Console │             │  Email   │             │ Webhook │
         │(testing)│             │ Channel  │             │ (HTTP)  │
         └─────────┘             └────┬─────┘             └─────────┘
                                      │ REST/WebSocket
                                      ▼
                          ┌───────────────────────┐
                          │  EMAIL-AGENT (Bun/TS) │
                          │  IMAP ←→ Gmail/etc    │
                          └───────────────────────┘
```

## Project Structure

```
hybrid-orchestrator/
├── core/
│   ├── __init__.py              # Main exports
│   ├── orchestrator/            # Central coordinator
│   │   └── orchestrator.py      # Orchestrator class
│   ├── storage/                 # Session persistence
│   │   ├── models.py            # Session, Activity dataclasses
│   │   └── store.py             # SQLite SessionStore
│   ├── triggers/                # Behavior detection
│   │   ├── models.py            # Trigger, Condition, Action
│   │   └── engine.py            # TriggerEngine evaluation
│   ├── channels/                # Communication routing
│   │   ├── base.py              # Channel ABC, Message, Recipient
│   │   ├── hub.py               # ChannelHub routing logic
│   │   ├── console.py           # Console output (testing)
│   │   ├── webhook.py           # HTTP webhook (with security)
│   │   ├── email.py             # Email via email-agent microservice
│   │   └── email_listener.py    # WebSocket listener for email events
│   └── agents/                  # AI decision-making
│       ├── base.py              # Agent ABC, AgentResponse
│       └── claude.py            # Claude + Mock implementations
├── config/
│   └── channels.yaml            # Channel configuration
├── tests/                       # 97 tests
│   ├── test_storage.py
│   ├── test_triggers.py
│   ├── test_channels.py
│   ├── test_email_channel.py    # Email channel tests (31 tests)
│   └── test_agents.py
├── examples/
│   ├── demo.py                  # Basic patterns demo
│   └── agent_demo.py            # AI integration demo
├── paper/
│   └── main.md                  # Research paper
├── patent/
│   └── provisional_draft.md     # Patent application
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Trigger Types

| Type | Description | Example |
|------|-------------|---------|
| `NO_ACTIVITY` | User inactive for N seconds | Detect abandonment |
| `FIELD_ERROR` | Same field changed N times | User struggling |
| `FIELD_CHANGED` | Specific field was changed | Track progress |
| `STATUS_CHANGED` | Session status matches value | State transitions |
| `CUSTOM` | Custom function returns true | Any logic |

## Channel Types

| Type | Description | Security |
|------|-------------|----------|
| `CONSOLE` | Print to stdout | Testing only |
| `WEBHOOK` | HTTP POST to URL | HTTPS required, SSRF protection |
| `EMAIL` | Email via email-agent | API key auth, microservice |
| `VOICE` | Voice call (VAPI) | Not implemented (proprietary) |
| `SMS` | Text message | Not implemented |
| `SLACK` | Slack message | Not implemented |
| `DASHBOARD` | Admin dashboard | Not implemented |

## Security

The webhook channel includes:
- **HTTPS required** by default (HTTP blocked)
- **Private IP blocking** - Prevents SSRF attacks (10.x, 192.168.x, 127.x blocked)
- **Domain allowlist** - Optional restriction to specific domains
- **Localhost blocking** - Can't send to localhost

```python
# Secure webhook configuration
config = ChannelConfig(
    type=ChannelType.WEBHOOK,
    config={
        "url": "https://api.example.com/webhook",
        "allowed_domains": ["*.example.com"],  # Only these domains
        "headers": {"Authorization": "Bearer token"},
    },
)
```

## Email Channel

The email channel integrates with the [email-agent](https://github.com/anthropics/claude-agent-sdk-demos/tree/main/email-agent) microservice for real email functionality.

### Setup

1. **Clone and configure email-agent:**
```bash
cd ~/dev/research
git clone https://github.com/anthropics/claude-agent-sdk-demos.git
cd claude-agent-sdk-demos/email-agent
cp .env.example .env
# Edit .env with your IMAP credentials and SERVICE_API_KEY
bun install
bun run start
```

2. **Use EmailChannel in Python:**
```python
from core.channels import EmailChannel, ChannelConfig, ChannelType

config = ChannelConfig(
    type=ChannelType.EMAIL,
    config={
        "base_url": "http://localhost:3000",
        "api_key": "your-service-api-key",  # Must match SERVICE_API_KEY in email-agent
        "timeout": 30,
    },
)

channel = EmailChannel(config)

# Register with orchestrator
orchestrator.channels.register(channel)

# Check health
healthy = await channel.check_health()

# Get inbox
emails = await channel.get_inbox(limit=20)
```

### Email Event Listener

Subscribe to real-time email events via WebSocket:

```python
from core.channels import EmailAgentListener, EmailEvent

async def handle_new_email(event: EmailEvent):
    print(f"New email from {event.sender}: {event.subject}")
    # Create session, trigger actions, etc.

listener = EmailAgentListener(
    ws_url="ws://localhost:3000/ws",
    api_key="your-service-api-key",
    on_email_received=handle_new_email,
)

await listener.start()
# ... listener runs in background ...
await listener.stop()
```

### Configuration

See `config/channels.yaml` for full configuration options including environment variable substitution.

## Limitations

1. **No controlled experiments** - We can't prove hybrid is better than alternatives
2. **Single case study** - Only validated in insurance domain
3. **SQLite only** - No PostgreSQL adapter yet
4. **English-only** - Not tested with other languages
5. **U.S.-focused** - Phone/SMS patterns differ internationally

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_triggers.py -v

# Run with coverage
pytest tests/ --cov=core
```

## Research Paper

See `paper/main.md`:

**"The Hybrid Orchestrator: A Framework for Coordinating Human-AI Teams"**

## License

Apache 2.0 - Use freely, but no warranty.

## Citation

```bibtex
@misc{sukhachev2026hybrid,
  title={The Hybrid Orchestrator: A Framework for Coordinating Human-AI Teams},
  author={Sukhachev, Pavel},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/pavelsukhachev/hybrid-orchestrator}
}
```

## Contact

Pavel Sukhachev
- Email: pavel@electromania.llc
- LinkedIn: linkedin.com/in/pavelsukhachev
- Company: Electromania LLC
