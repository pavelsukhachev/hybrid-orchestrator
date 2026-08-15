/* Hybrid Orchestrator Demo 0 — mock router. No network except same-origin JSON. */
(function () {
  "use strict";

  var TYPE_MS = 2000;
  var state = {
    pack: null,
    currentId: null,
    timer: null,
    running: false
  };

  function $(id) { return document.getElementById(id); }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function pretty(value) {
    return String(value || "").replace(/_/g, " ");
  }

  function policyShort(type) {
    var map = {
      commercial_general_liability: "CGL",
      personal_auto: "Personal auto",
      homeowners: "Homeowners"
    };
    return map[type] || pretty(type);
  }

  function findScenario(id) {
    var list = state.pack.scenarios;
    for (var i = 0; i < list.length; i++) {
      if (list[i].id === id) return list[i];
    }
    return list[0];
  }

  function mappingFor(routing) {
    var map = state.pack.library_mapping || {};
    return map[routing] || { hf_label: routing, library: "", agent_api: "" };
  }

  function setStatus(label, mode) {
    var el = $("run-status");
    el.textContent = label;
    el.setAttribute("data-state", mode || "idle");
  }

  function renderTabs() {
    var host = $("scenario-tabs");
    host.innerHTML = "";
    state.pack.scenarios.forEach(function (sc) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tab";
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", sc.id === state.currentId ? "true" : "false");
      btn.dataset.id = sc.id;
      btn.innerHTML = esc(sc.demo.short_label) +
        '<span class="hint">' + esc(sc.demo.contrast_label) + "</span>";
      btn.addEventListener("click", function () {
        if (state.running) return;
        selectScenario(sc.id, true);
      });
      host.appendChild(btn);
    });
  }

  function renderMeta(sc) {
    var ctx = sc.input.context;
    var bits = [
      { t: policyShort(ctx.policy_type), c: "policy" },
      { t: ctx.policy_status, c: "" },
      { t: ctx.customer_tier, c: ctx.customer_tier === "high_value" ? "gold" : "" },
      { t: ctx.channel, c: "" },
      { t: ctx.time_of_day, c: "" },
      { t: ctx.state, c: "" },
      { t: sc.difficulty, c: sc.difficulty === "hard" ? "warn" : "ok" },
      { t: pretty(sc.subcategory), c: "" }
    ];
    $("intake-meta").innerHTML = bits.map(function (b) {
      return '<span class="chip ' + b.c + '">' + esc(pretty(b.t)) + "</span>";
    }).join("");
    $("scenario-id").textContent = sc.id;
    $("intake-who").textContent =
      pretty(ctx.channel) + " · " + pretty(ctx.time_of_day) + " · " + ctx.state;
  }

  function idleBoard() {
    $("board-body").innerHTML =
      '<div class="decision-idle" id="board-idle">' +
      '<span class="idle-mark">—</span>' +
      "Waiting for intake. Routing stays empty until the mock agent reads the message." +
      "</div>";
  }

  function idleQueue() {
    $("queue-body").innerHTML =
      '<div class="queue-idle" id="queue-idle">' +
      '<span class="idle-mark">∅</span>' +
      "No handoff yet. Routine claims never reach this desk." +
      "</div>";
    $("queue-count").textContent = "0 waiting";
  }

  function idleIntake() {
    $("intake-text").innerHTML =
      '<span class="placeholder">Select Run to take the first notice of loss.</span>';
  }

  function renderBoard(sc) {
    var out = sc.expected_output;
    var routing = out.routing_decision;
    var map = mappingFor(routing);
    var lib = sc.demo.library_action || map.library;
    var chipClass = routing === "ai_handle" ? "handle" : "handoff";
    var flags = out.compliance_flags || [];
    var flagHtml = flags.length
      ? flags.map(function (f) {
          return '<span class="chip warn">' + esc(f) + "</span>";
        }).join(" ")
      : '<span class="chip ok">none</span>';

    var actions = (out.required_actions || []).map(function (a) {
      return "<li><span class=\"tick\" aria-hidden=\"true\">✓</span>" + esc(a) + "</li>";
    }).join("");

    var trace = (sc.demo.router_trace || []).map(function (row) {
      var cls = "ok";
      if (row.result === "flag" || row.result === "prohibited" || row.result === "none" && /authority|fault/i.test(row.rule)) {
        cls = row.result === "flag" || row.result === "prohibited" ? "flag" : "ok";
      }
      if (row.result === "none" && /Litigation/i.test(row.rule)) cls = "ok";
      if (row.result === "none" && /authority/i.test(row.rule)) cls = "flag";
      return '<div class="trace-row"><span>' + esc(row.rule) +
        " <span style=\"color:var(--muted)\">· " + esc(row.detail) +
        '</span></span><span class="trace-result ' + cls + '">' +
        esc(row.result) + "</span></div>";
    }).join("");

    $("board-body").innerHTML =
      '<div class="reveal">' +
        '<div class="route-row">' +
          '<div class="hf-chip ' + chipClass + '">' +
            '<span class="hf">' + esc(routing) + "</span>" +
            '<span class="lib">library ' + esc(lib) + "</span>" +
          "</div>" +
          '<span class="chip">' + esc(sc.demo.library_note) + "</span>" +
        "</div>" +
        '<dl class="field"><dt>Intent</dt><dd><code>' + esc(out.intent) + "</code></dd></dl>" +
        '<dl class="field"><dt>Priority</dt><dd>' + esc(out.priority) + "</dd></dl>" +
        '<dl class="field"><dt>Compliance</dt><dd>' + flagHtml + "</dd></dl>" +
        '<div class="why"><strong>Why</strong>' + esc(sc.demo.why) + "</div>" +
        "<ul class=\"actions-list\">" + actions + "</ul>" +
        '<div class="trace"><h3>Mock router</h3>' + trace + "</div>" +
      "</div>";
  }

  function renderQueue(sc) {
    var q = sc.demo.queue;
    var out = sc.expected_output;
    var ctx = sc.input.context;

    if (q.kind === "empty") {
      $("queue-body").innerHTML =
        '<div class="queue-card empty-ok reveal">' +
          "<div class=\"queue-top\">" +
            "<h3 class=\"queue-title\">Queue empty</h3>" +
            "<span class=\"priority standard\">ai_handle</span>" +
          "</div>" +
          "<p class=\"handoff-note\">" + esc(q.handoff_note) + "</p>" +
        "</div>";
      $("queue-count").textContent = "0 waiting";
      return;
    }

    var kindClass = q.kind === "adjuster" ? "adjuster" : "";
    var instr = (q.instructions || []).map(function (i) {
      return "<li>" + esc(i) + "</li>";
    }).join("");

    $("queue-body").innerHTML =
      '<div class="queue-card ' + kindClass + ' reveal">' +
        '<div class="queue-top">' +
          '<div><h3 class="queue-title">' + esc(q.title) + "</h3>" +
          '<div class="chip" style="margin-top:6px">' + esc(q.assignee) + "</div></div>" +
          '<span class="priority">' + esc(out.priority) + "</span>" +
        "</div>" +
        "<ul class=\"instr\">" + instr + "</ul>" +
        '<div class="packet">' +
          "<h3>Context packet — do not re-ask</h3>" +
          '<dl class="field"><dt>Incident</dt><dd>' + esc(sc.input.customer_message) + "</dd></dl>" +
          '<dl class="field"><dt>Policy</dt><dd>' +
            esc(policyShort(ctx.policy_type) + " · " + ctx.policy_status + " · " +
              ctx.customer_tier + " · " + ctx.state) + "</dd></dl>" +
          '<dl class="field"><dt>Flags</dt><dd>' +
            esc((out.compliance_flags || []).join(", ") || "none") + "</dd></dl>" +
          '<dl class="field"><dt>AI actions</dt><dd>' +
            esc((out.required_actions || []).join(" · ")) + "</dd></dl>" +
        "</div>" +
        "<p class=\"handoff-note\">" + esc(q.handoff_note) + "</p>" +
      "</div>";
    $("queue-count").textContent = "1 waiting";
  }

  function stopTimer() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  function typeIntake(sc, done) {
    var msg = sc.input.customer_message;
    var host = $("intake-text");
    var i = 0;
    var step = Math.max(18, Math.floor(TYPE_MS / Math.max(msg.length, 1)));
    host.innerHTML = '<span id="typed"></span><span class="caret" aria-hidden="true"></span>';
    var typed = $("typed");
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      typed.textContent = msg;
      done();
      return;
    }
    state.timer = setInterval(function () {
      i += 1;
      typed.textContent = msg.slice(0, i);
      if (i >= msg.length) {
        stopTimer();
        done();
      }
    }, step);
  }

  function selectScenario(id, resetView) {
    state.currentId = id;
    var sc = findScenario(id);
    renderTabs();
    renderMeta(sc);
    if (resetView) {
      stopTimer();
      state.running = false;
      $("run-btn").disabled = false;
      setStatus("Idle", "idle");
      idleIntake();
      idleBoard();
      idleQueue();
    }
  }

  function run() {
    if (state.running || !state.pack) return;
    var sc = findScenario(state.currentId);
    state.running = true;
    $("run-btn").disabled = true;
    idleBoard();
    idleQueue();
    setStatus("Listening", "running");

    typeIntake(sc, function () {
      setStatus("Routing", "running");
      window.setTimeout(function () {
        renderBoard(sc);
        var routing = sc.expected_output.routing_decision;
        if (routing === "ai_handle") {
          setStatus("AI handle", "handled");
        } else {
          setStatus("Escalated", "escalated");
        }
        window.setTimeout(function () {
          renderQueue(sc);
          state.running = false;
          $("run-btn").disabled = false;
        }, 180);
      }, 220);
    });
  }

  function reset() {
    stopTimer();
    state.running = false;
    $("run-btn").disabled = false;
    selectScenario(state.currentId, true);
  }

  function attach() {
    $("run-btn").addEventListener("click", run);
    $("reset-btn").addEventListener("click", reset);
  }

  function boot(pack) {
    state.pack = pack;
    var def = pack.default_scenario || pack.scenarios[0].id;
    attach();
    selectScenario(def, true);
  }

  function parseEmbedded() {
    var el = $("embedded-scenarios");
    if (!el) throw new Error("No embedded scenarios");
    return JSON.parse(el.textContent);
  }

  function start() {
    fetch("scenarios.json", { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("scenarios.json " + res.status);
        return res.json();
      })
      .then(boot)
      .catch(function () {
        boot(parseEmbedded());
      });
  }

  start();
})();
