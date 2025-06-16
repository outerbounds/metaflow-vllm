from metaflow.cards import Markdown, Table, VegaChart
from metaflow.metaflow_current import current
from datetime import datetime
import threading
import time


from metaflow.exception import MetaflowException
from collections import defaultdict


class CardDecoratorInjector:
    """
    Mixin Useful for injecting @card decorators from other first class Metaflow decorators.
    """

    _first_time_init = defaultdict(dict)

    @classmethod
    def _get_first_time_init_cached_value(cls, step_name, card_id):
        return cls._first_time_init.get(step_name, {}).get(card_id, None)

    @classmethod
    def _set_first_time_init_cached_value(cls, step_name, card_id, value):
        cls._first_time_init[step_name][card_id] = value

    def _card_deco_already_attached(self, step, card_id):
        for decorator in step.decorators:
            if decorator.name == "card":
                if decorator.attributes["id"] and card_id == decorator.attributes["id"]:
                    return True
        return False

    def _get_step(self, flow, step_name):
        for step in flow:
            if step.name == step_name:
                return step
        return None

    def _first_time_init_check(self, step_dag_node, card_id):
        """ """
        return not self._card_deco_already_attached(step_dag_node, card_id)

    def attach_card_decorator(
        self,
        flow,
        step_name,
        card_id,
        card_type,
        refresh_interval=5,
    ):
        """
        This method is called `step_init` in your StepDecorator code since
        this class is used as a Mixin
        """
        from metaflow import decorators as _decorators

        if not all([card_id, card_type]):
            raise MetaflowException(
                "`INJECTED_CARD_ID` and `INJECTED_CARD_TYPE` must be set in the `CardDecoratorInjector` Mixin"
            )

        step_dag_node = self._get_step(flow, step_name)
        if (
            self._get_first_time_init_cached_value(step_name, card_id) is None
        ):  # First check class level setting.
            if self._first_time_init_check(step_dag_node, card_id):
                self._set_first_time_init_cached_value(step_name, card_id, True)
                _decorators._attach_decorators_to_step(
                    step_dag_node,
                    [
                        "card:type=%s,id=%s,refresh_interval=%s"
                        % (card_type, card_id, str(refresh_interval))
                    ],
                )
            else:
                self._set_first_time_init_cached_value(step_name, card_id, False)


class CardRefresher:

    CARD_ID = None

    def on_startup(self, current_card):
        raise NotImplementedError("make_card method must be implemented")

    def on_error(self, current_card, error_message):
        raise NotImplementedError("error_card method must be implemented")

    def on_update(self, current_card, data_object):
        raise NotImplementedError("update_card method must be implemented")

    def sqlite_fetch_func(self, conn):
        raise NotImplementedError("sqlite_fetch_func must be implemented")


class VLLMStatusCard(CardRefresher):
    """
    Real-time status card for vLLM system monitoring.
    Shows server health, model status, and recent events.

    Intended to be inherited from in a step decorator like this:
        class VLLMDecorator(StepDecorator, VLLMStatusCard):
    """

    CARD_ID = "vllm_status"

    def __init__(self, refresh_interval=10):
        self.refresh_interval = refresh_interval
        self.status_data = {
            "server": {
                "status": "Starting",
                "uptime_start": None,
                "last_health_check": None,
                "health_status": "Unknown",
                "models": [],
            },
            "models": {},  # model_name -> {status, load_time, etc}
            "performance": {
                "install_time": None,
                "server_startup_time": None,
                "total_initialization_time": None,
            },
            "versions": {
                "vllm": "Detecting...",
            },
            "events": [],  # Recent events log
        }
        self._lock = threading.Lock()
        self._already_rendered = False

    def update_status(self, category, data):
        """Thread-safe method to update status data"""
        with self._lock:
            if category in self.status_data:
                self.status_data[category].update(data)

    def add_event(self, event_type, message, timestamp=None):
        """Add an event to the timeline"""
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self.status_data["events"].insert(
                0,
                {
                    "type": event_type,  # 'info', 'warning', 'error', 'success'
                    "message": message,
                    "timestamp": timestamp,
                },
            )
            # Keep only last 10 events
            self.status_data["events"] = self.status_data["events"][:10]

    def get_circuit_breaker_emoji(self, state):
        """Get status emoji for circuit breaker state"""
        emoji_map = {"CLOSED": "🟢", "OPEN": "🔴", "HALF_OPEN": "🟡"}
        return emoji_map.get(state, "⚪")

    def get_uptime_string(self, start_time):
        """Calculate uptime string"""
        if not start_time:
            return "Not started"

        uptime = datetime.now() - start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def on_startup(self, current_card):
        """Initialize the card when monitoring starts"""
        current_card.append(Markdown("# 🚀 `@vllm` Status Dashboard"))
        current_card.append(Markdown("_Initializing vLLM system..._"))
        current_card.refresh()

    def render_card_fresh(self, current_card, data):
        """Render the complete card with all status information"""
        self._already_rendered = True
        current_card.clear()

        current_card.append(Markdown("# 🚀 `@vllm` Status Dashboard"))

        versions = data.get("versions", {})
        vllm_version = versions.get("vllm", "Unknown")
        current_card.append(
            Markdown(f"**vLLM Version:** `{vllm_version}`")
        )

        current_card.append(
            Markdown(f"_Last updated: {datetime.now().strftime('%H:%M:%S')}_")
        )

        server_data = data["server"]
        uptime = self.get_uptime_string(server_data.get("uptime_start"))
        server_status = server_data.get("status", "Unknown")
        model = server_data.get("model", "Unknown")

        # Determine status emoji
        if server_status == "Running":
            status_emoji = "🟢"
            model_emoji = "✅"
        elif server_status == "Failed":
            status_emoji = "🔴"
            model_emoji = "❌"
        elif server_status == "Starting":
            status_emoji = "🟡"
            model_emoji = "⏳"
        else:  # Stopped, etc.
            status_emoji = "⚫"
            model_emoji = "⏹️"

        # Main status section
        current_card.append(
            Markdown(f"## {status_emoji} Server Status: {server_status}")
        )

        if server_status == "Running" and uptime:
            current_card.append(Markdown(f"**Uptime:** {uptime}"))
        
        # Model information - only show detailed status if server is running
        if server_status == "Running":
            current_card.append(Markdown(f"## {model_emoji} Model: `{model}`"))
            
            # Show model-specific status if available
            models_data = data.get("models", {})
            if models_data and model in models_data:
                model_info = models_data[model]
                model_status = model_info.get("status", "Unknown")
                load_time = model_info.get("load_time")
                
                current_card.append(Markdown(f"**Status:** {model_status}"))
                if load_time and isinstance(load_time, (int, float)):
                    current_card.append(Markdown(f"**Load Time:** {load_time:.1f}s"))
        elif model != "Unknown":
            current_card.append(Markdown(f"## {model_emoji} Model: `{model}` (Server Stopped)"))
        
        # Simplified monitoring note
        current_card.append(
            Markdown("## 🔧 Monitoring\n**Advanced Features:** Disabled (Circuit Breaker, Request Interception)")
        )

        # Performance metrics
        perf_data = data["performance"]
        if any(v is not None for v in perf_data.values()):
            current_card.append(Markdown("## ⚡ Performance"))

            init_metrics = []
            shutdown_metrics = []

            for metric, value in perf_data.items():
                if value is not None:
                    display_value = f"{value:.1f}s" if isinstance(value, (int, float)) else value
                    metric_display = metric.replace("_", " ").title()

                    if "shutdown" in metric.lower():
                        shutdown_metrics.append([metric_display, display_value])
                    elif metric in ["install_time", "server_startup_time", "total_initialization_time"]:
                        init_metrics.append([metric_display, display_value])

            if init_metrics:
                current_card.append(Markdown("### Initialization"))
                current_card.append(Table(init_metrics, headers=["Metric", "Duration"]))

            if shutdown_metrics:
                current_card.append(Markdown("### Shutdown"))
                current_card.append(Table(shutdown_metrics, headers=["Metric", "Value"]))

        # Recent events
        events = data.get("events", [])
        if events:
            current_card.append(Markdown("## 📝 Recent Events"))
            for event in events[:5]:  # Show last 5 events
                event_type = event.get("type", "info")
                message = event.get("message", "")
                timestamp = event.get("timestamp", datetime.now())
                
                emoji_map = {
                    "info": "ℹ️",
                    "success": "✅", 
                    "warning": "⚠️",
                    "error": "❌"
                }
                emoji = emoji_map.get(event_type, "ℹ️")
                
                time_str = timestamp.strftime("%H:%M:%S") if isinstance(timestamp, datetime) else str(timestamp)
                current_card.append(Markdown(f"- {emoji} `{time_str}` {message}"))

        current_card.refresh()

    def on_error(self, current_card, error_message):
        """Handle errors in card rendering"""
        if not self._already_rendered:
            current_card.clear()
            current_card.append(Markdown("# 🚀 `@vllm` Status Dashboard"))
            current_card.append(Markdown(f"## ❌ Error: {str(error_message)}"))
            current_card.refresh()

    def on_update(self, current_card, data_object):
        """Update the card with new data"""
        with self._lock:
            current_data = self.status_data.copy()

        if not self._already_rendered:
            self.render_card_fresh(current_card, current_data)
        else:
            # For frequent updates, we could implement incremental updates here
            # For now, just re-render the whole card
            self.render_card_fresh(current_card, current_data)

    def sqlite_fetch_func(self, conn):
        """Required by CardRefresher (which needs a refactor), but we use in-memory data instead"""
        with self._lock:
            return {"status": self.status_data}
