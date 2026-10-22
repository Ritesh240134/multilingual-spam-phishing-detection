/* Set apiBaseUrl before app.js loads when using a non-default API origin. */
window.SIGNALCHECK_CONFIG = {
  apiBaseUrl: window.SIGNALCHECK_CONFIG?.apiBaseUrl || "http://127.0.0.1:8000",
};
