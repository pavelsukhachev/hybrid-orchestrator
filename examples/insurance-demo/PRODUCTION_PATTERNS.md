# Production Orchestration Patterns

Patterns extracted from a production voice AI system for insurance to demonstrate hybrid human-AI orchestration.

Interactive Demo 0 lives in `/docs`.

## System Overview

A voice-guided insurance application platform where:
- **AI Agent**: Guides customers through forms via phone
- **Human Agents**: Handle exceptions and monitor progress
- **Backend**: Coordinates between voice AI and external systems
- **Dashboard**: Provides real-time visibility for supervisors

## Key Patterns

### 1. Voice-to-Web Orchestration

```
Customer Call → Voice AI → Backend Webhook → External APIs → Web Form
                  ↑                              ↓
              Tool Results ← ← ← ← ← Screen State Updates
```

- Voice AI calls tools (`startSession`, `sendSms`, `getScreenState`, `highlightField`)
- Backend handles tool calls via webhook, returns structured results
- Voice AI interprets results and guides customer

### 2. Session State Management

**Pattern**: Use database as source of truth for cross-session continuity

```python
# Pseudocode
async def get_or_create_session(call_id, customer_info):
    return db.session.upsert(
        where={"call_id": call_id},
        update={...},
        create={...}
    )
```

**Key Design**:
- Each phone call gets a unique session
- Session stores: customer info, auth token, control URL
- Sessions expire after 24 hours

### 3. Screen State Polling

**Pattern**: Backend tracks user's web form progress via webhook from form system

```python
# Pseudocode
async def get_screen_state(session_token):
    # Get latest screen activities
    activities = await db.screen_activity.find_many(
        where={"session_token": session_token},
        order_by={"created_at": "desc"},
        take=2  # Current + previous for change detection
    )

    # Detect page from screen name or field IDs
    page_number = get_page_number(activity.screen_name, activity.fields)

    # Detect field value changes
    changed_fields = detect_changes(previous.fields, latest.fields)

    return {
        "screen_name": screen_name,
        "page_number": page_number,
        "fields": fields,
        "changed_fields": changed_fields,
        "data_age": {"seconds_since_update": delta, "is_fresh": delta < 10}
    }
```

**Key Design**:
- Form system pushes screen updates to backend
- Backend stores in activity table
- Voice AI polls for latest state
- Change detection enables "I see you changed X to Y"

### 4. Field Highlighting (Remote UI Control)

**Pattern**: Backend queues UI commands, client polls and executes

```python
# Pseudocode
async def highlight_field(session_token, field_id):
    pending = {"action": "highlighted", "field_id": field_id}
    await db.session.update(
        where={"token": session_token},
        data={"pending_highlight": pending}
    )
    return {"success": True}
```

**Key Design**:
- Voice AI requests highlight
- Backend stores in pending queue
- Web client polls, receives command, clears queue
- Voice AI can verify delivery via subsequent state check

### 5. Error Handling for Voice AI

**Pattern**: Always return 200 to voice platform, include error in payload

```python
# Pseudocode
try:
    result = await process_tool_call(request)
    return Response(status=200, body=result)
except Exception as e:
    # CRITICAL: Always return 200
    # Voice platforms fail hard on HTTP errors
    return Response(status=200, body={
        "success": False,
        "error": "Temporary issue. Please try again."
    })
```

**Key Design**:
- Voice platforms fail hard on HTTP errors
- Return 200 with error in payload instead
- Voice AI can retry or inform customer gracefully

### 6. AI Prompt Engineering

**Pattern**: Explicit rules with banned behaviors

```
System prompt structure:
1. Role definition (friendly, short, never pause)
2. Step-by-step workflow (collect info → send link → guide)
3. Explicit banned phrases ("unknown", "none", "let me know when")
4. Examples of desired behavior
5. ALL CAPS for critical rules
```

**Key Design**:
- Explicit step-by-step workflow
- Banned phrases prevent common AI mistakes
- Examples of desired behavior
- All caps for critical rules

### 7. Tool Definitions with Status Messages

**Pattern**: Tools have built-in status messages for natural conversation

```python
tools = [{
    "name": "startSession",
    "messages": {
        "request_start": "Setting that up...",
        "request_complete": "",  # Silent on success
        "request_failed": "Let me try that again..."
    }
}]
```

**Key Design**:
- `request_start`: Spoken while tool executes
- `request_complete`: Spoken on success (empty = silent)
- `request_failed`: Spoken on error
- Keeps conversation natural during async operations

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INSURANCE AI PLATFORM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     │
│  │   CUSTOMER    │     │   VOICE AI    │     │    BACKEND    │     │
│  │   (Phone)     │────▶│   (Agent)     │────▶│   (Server)    │     │
│  │               │◀────│               │◀────│               │     │
│  └───────────────┘     └───────────────┘     └───────────────┘     │
│         │                     │                      │              │
│         │                     │              ┌───────┴───────┐      │
│         │                     │              │               │      │
│         ▼                     │              ▼               ▼      │
│  ┌───────────────┐           │       ┌───────────┐   ┌───────────┐ │
│  │  WEB FORM     │           │       │ FORM API  │   │  DATABASE │ │
│  │  (Browser)    │◀──────────┼──────▶│           │   │           │ │
│  └───────────────┘           │       └───────────┘   └───────────┘ │
│         │                    │                              │       │
│         └────────────────────┼──────────────────────────────┘       │
│                              │                                      │
│  ┌───────────────┐           │       ┌───────────────┐             │
│  │  HUMAN AGENT  │◀──────────┴──────▶│   DASHBOARD   │             │
│  │  (Exception)  │                   │   (Web App)   │             │
│  └───────────────┘                   └───────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Customer calls** → Voice AI receives, runs agent
2. **Agent collects info** → Calls `startSession` tool
3. **Backend creates session** → Form API, stores token
4. **Agent sends SMS** → Calls `sendSms` tool
5. **Customer opens form** → Form system pushes screen updates to backend
6. **Agent guides** → Calls `getScreenState` to see progress
7. **Agent highlights** → Calls `highlightField` for visual guidance
8. **Human monitors** → Dashboard shows all sessions in real-time
9. **Exception handling** → Human agent takes over if needed

## Lessons Learned

1. **Always return 200 to voice platforms** - HTTP errors break AI completely
2. **Screen name detection is unreliable** - Use field IDs as fallback
3. **Explicit prompts work better** - Ban unwanted phrases explicitly
4. **Database timeouts are essential** - Prevent hanging during load
5. **Change detection enables natural dialogue** - "I see you changed X"
6. **Tool status messages keep conversation flowing** - No awkward silences

## Relevance to Hybrid Orchestrator

This production system demonstrates key concepts from the Hybrid Orchestrator framework:

| Concept | Production Implementation |
|---------|------------------------|
| **Hybrid Team** | AI Agent + Human Agents + Customer |
| **Omnichannel** | Voice + SMS + Web Form |
| **Task Tracking** | Database sessions + Dashboard |
| **Blocker Detection** | Screen state polling + page progress |
| **Proactive Intervention** | Highlighting + verbal guidance |
| **Human Escalation** | Dashboard monitoring + notifications |

This production system validates the core thesis: hybrid human-AI teams with multi-channel communication outperform pure AI or pure human approaches.
