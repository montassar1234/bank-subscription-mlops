import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function formatPercent(value, digits = 1) {
  return `${Number(value || 0).toFixed(digits)}%`;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(Number(value || 0));
}

function normalizeLabel(label) {
  return label.replaceAll("_", " ");
}

function compactAction(text) {
  if (!text) {
    return "Review this customer in the next campaign wave.";
  }
  return text;
}

export default function App() {
  const [overview, setOverview] = useState(null);
  const [explainability, setExplainability] = useState(null);
  const [form, setForm] = useState(null);
  const [simulatorForm, setSimulatorForm] = useState(null);
  const [result, setResult] = useState(null);
  const [whatIfResult, setWhatIfResult] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [batchFile, setBatchFile] = useState(null);
  const [batchPage, setBatchPage] = useState(1);
  const [batchPageSize, setBatchPageSize] = useState(20);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [simulatorLoading, setSimulatorLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        const [overviewResponse, explainabilityResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/dashboard/overview`),
          fetch(`${API_BASE_URL}/dashboard/explainability`)
        ]);

        const overviewData = await overviewResponse.json();
        const explainabilityData = await explainabilityResponse.json();

        if (!overviewResponse.ok) {
          throw new Error(overviewData.detail || "Unable to load dashboard data.");
        }

        if (!explainabilityResponse.ok) {
          throw new Error(explainabilityData.detail || "Unable to load model insights.");
        }

        if (!cancelled) {
          setOverview(overviewData);
          setExplainability(explainabilityData);
          setForm(overviewData.feature_schema.default_form);
          setSimulatorForm(overviewData.feature_schema.default_form);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, []);

  const featureSchema = overview?.feature_schema;
  const categoricalOptions = featureSchema?.categorical_options || {};
  const numericFields = featureSchema?.numeric_fields || [];
  const formFields = useMemo(() => (form ? Object.keys(form) : []), [form]);
  const simulatorFields = ["contact", "campaign", "poutcome", "balance", "housing", "month"];

  const updateField = (setter) => (field, value) => {
    setter((prev) => ({
      ...prev,
      [field]: numericFields.includes(field) ? Number(value) : value
    }));
  };

  const updatePredictionField = updateField(setForm);
  const updateSimulatorField = updateField(setSimulatorForm);

  const submitPrediction = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const submitBatch = async () => {
    if (!batchFile) {
      return;
    }

    setBatchLoading(true);
    setError("");
    setBatchResult(null);
    setBatchPage(1);

    try {
      const payload = new FormData();
      payload.append("file", batchFile);

      const response = await fetch(`${API_BASE_URL}/predict_batch`, {
        method: "POST",
        body: payload
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Batch prediction failed.");
      }
      setBatchResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBatchLoading(false);
    }
  };

  const runWhatIfSimulation = async () => {
    if (!overview?.feature_schema?.default_form || !simulatorForm) {
      return;
    }

    setSimulatorLoading(true);
    setError("");
    setWhatIfResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/what-if`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline: overview.feature_schema.default_form,
          scenario: simulatorForm
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Scenario analysis failed.");
      }
      setWhatIfResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSimulatorLoading(false);
    }
  };

  const predictionProbability = result ? Number(result.probability_yes) : 0;
  const leaderboard = overview?.model_leaderboard || [];
  const segments = overview?.segment_insights || [];
  const dataset = overview?.dataset;
  const governance = overview?.model_governance;
  const topFeatures = explainability?.top_features || [];
  const strategicRecommendations = explainability?.strategic_recommendations || [];
  const bestSegment = segments[0];
  const batchRecords = batchResult?.records || [];
  const totalBatchPages = Math.max(1, Math.ceil(batchRecords.length / batchPageSize));
  const currentBatchPage = Math.min(batchPage, totalBatchPages);
  const batchStartIndex = (currentBatchPage - 1) * batchPageSize;
  const paginatedBatchRecords = batchRecords.slice(batchStartIndex, batchStartIndex + batchPageSize);

  return (
    <main className="page-shell">
      <section className="hero-panel hero-panel-business">
        <div className="hero-copy">
          <p className="eyebrow">Banking Decision Intelligence</p>
          <h1>Subscription Campaign Command Center</h1>
          <p className="hero-subtitle">
            Score customers, identify high-potential segments, and optimize outreach strategy from a
            single business-ready workspace.
          </p>

          <div className="hero-actions">
            <a href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
              API Documentation
            </a>
            <a href="http://127.0.0.1:5001" target="_blank" rel="noreferrer">
              Model Tracking
            </a>
          </div>

          <div className="hero-inline-note">
            <strong>{governance?.feature_set || "Pre-contact scoring"}</strong>
            <p>{governance?.leakage_note}</p>
          </div>
        </div>

        <div className="hero-stat-grid">
          <article className="stat-card stat-accent">
            <span>Champion model</span>
            <strong>{governance?.best_model || "Loading..."}</strong>
            <small>Active scoring pipeline</small>
          </article>
          <article className="stat-card">
            <span>Model quality</span>
            <strong>{formatPercent(governance?.best_f1 || 0)}</strong>
            <small>F1 score</small>
          </article>
          <article className="stat-card">
            <span>API status</span>
            <strong>{governance?.api_status || "Checking..."}</strong>
            <small>Operational scoring service</small>
          </article>
          <article className="stat-card">
            <span>Reachable segment</span>
            <strong>{bestSegment?.segment || "Loading..."}</strong>
            <small>{bestSegment ? `${formatPercent(bestSegment.conversion_rate)} conversion` : "Segment insight"}</small>
          </article>
        </div>
      </section>

      {error && <section className="alert-card">{error}</section>}

      <section className="dashboard-strip">
        <article className="overview-card">
          <span>Customers in scope</span>
          <strong>{formatNumber(dataset?.records)}</strong>
          <p>Records available for scoring and segmentation</p>
        </article>
        <article className="overview-card">
          <span>Expected conversion</span>
          <strong>{formatPercent(dataset?.subscription_rate || 0)}</strong>
          <p>Baseline subscription rate across the full portfolio</p>
        </article>
        <article className="overview-card">
          <span>Median balance</span>
          <strong>${formatNumber(dataset?.median_balance)}</strong>
          <p>Financial capacity reference for campaign targeting</p>
        </article>
        <article className="overview-card">
          <span>Campaign pressure</span>
          <strong>{Number(dataset?.campaign_pressure_mean || 0).toFixed(2)}</strong>
          <p>Average contacts per campaign flow</p>
        </article>
      </section>

      <section className="workspace-grid">
        <article className="panel-card">
          <div className="panel-header">
            <h2>Portfolio Opportunities</h2>
            <p>Commercial segments with the strongest conversion potential</p>
          </div>

          <div className="segment-list">
            {segments.map((segment) => (
              <div className="segment-card" key={segment.dimension}>
                <span>{normalizeLabel(segment.dimension)}</span>
                <strong>{segment.segment}</strong>
                <p>
                  {formatPercent(segment.conversion_rate)} conversion rate across{" "}
                  {formatNumber(segment.sample_size)} customers
                </p>
              </div>
            ))}
          </div>
        </article>

        <article className="panel-card">
          <div className="panel-header">
            <h2>Campaign Actions</h2>
            <p>Recommended operational levers from model explainability</p>
          </div>

          <div className="action-list">
            {strategicRecommendations.slice(0, 3).map((item) => (
              <div className="action-card" key={item.title}>
                <div>
                  <span>{item.title}</span>
                  <strong>{(item.scenario_probability_yes * 100).toFixed(1)}%</strong>
                </div>
                <p>Potential uplift: {(item.uplift * 100).toFixed(1)} pts</p>
              </div>
            ))}
          </div>

          <div className="leaderboard-compact">
            {leaderboard.slice(0, 3).map((item) => (
              <div className={`leaderboard-row ${item.is_best ? "best-row" : ""}`} key={item.model}>
                <div>
                  <span className="row-label">{item.is_best ? "Champion" : "Alternative"}</span>
                  <strong>{item.model}</strong>
                </div>
                <div className="metric-stack">
                  <span>F1 {formatPercent(item.f1)}</span>
                  <span>ROC AUC {formatPercent(item.roc_auc)}</span>
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="workspace-grid workspace-grid-scoring">
        <article className="panel-card">
          <div className="panel-header">
            <h2>Customer Scoring</h2>
            <p>Evaluate a single client profile and surface the right commercial action</p>
          </div>

          {form && (
            <form className="form-grid" onSubmit={submitPrediction}>
              {formFields.map((field) => (
                <label key={field} className="field-card">
                  <span>{normalizeLabel(field)}</span>
                  {categoricalOptions[field] ? (
                    <select value={form[field]} onChange={(event) => updatePredictionField(field, event.target.value)}>
                      {categoricalOptions[field].map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      value={form[field]}
                      onChange={(event) => updatePredictionField(field, event.target.value)}
                    />
                  )}
                </label>
              ))}

              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "Scoring..." : "Score customer"}
              </button>
            </form>
          )}
        </article>

        <article className="panel-card result-panel">
          <div className="panel-header">
            <h2>Decision Summary</h2>
            <p>Probability, propensity band, and next best action</p>
          </div>

          {result ? (
            <div className="result-shell">
              <div className="score-ring">
                <div
                  className="score-ring-fill"
                  style={{ "--score": `${Math.round(predictionProbability * 100)}%` }}
                />
                <div className="score-ring-inner">
                  <strong>{(predictionProbability * 100).toFixed(1)}%</strong>
                  <span>Subscription probability</span>
                </div>
              </div>

              <div className="result-copy">
                <span className={`pill ${result.label === "yes" ? "pill-success" : "pill-muted"}`}>
                  {result.label === "yes" ? "Priority lead" : "Monitor"}
                </span>
                <h3>{result.risk_band}</h3>
                <p>{compactAction(result.recommended_action)}</p>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <strong>No customer scored yet</strong>
              <p>Use the form to generate a business-ready recommendation for an individual client.</p>
            </div>
          )}
        </article>
      </section>

      <section className="workspace-grid">
        <article className="panel-card">
          <div className="panel-header">
            <h2>Scenario Planner</h2>
            <p>Adjust a few controllable levers and compare the likely outcome before launch</p>
          </div>

          {simulatorForm && (
            <div className="simulator-grid">
              {simulatorFields.map((field) => (
                <label key={field} className="field-card">
                  <span>{normalizeLabel(field)}</span>
                  {categoricalOptions[field] ? (
                    <select
                      value={simulatorForm[field]}
                      onChange={(event) => updateSimulatorField(field, event.target.value)}
                    >
                      {categoricalOptions[field].map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      value={simulatorForm[field]}
                      onChange={(event) => updateSimulatorField(field, event.target.value)}
                    />
                  )}
                </label>
              ))}
            </div>
          )}

          <button className="secondary-button" onClick={runWhatIfSimulation} disabled={simulatorLoading}>
            {simulatorLoading ? "Analyzing..." : "Run scenario"}
          </button>

          {whatIfResult && (
            <div className="whatif-shell">
              <div className="whatif-grid">
                <div className="mini-card">
                  <span>Current plan</span>
                  <strong>{(whatIfResult.baseline.probability_yes * 100).toFixed(1)}%</strong>
                  <p>{whatIfResult.baseline.risk_band}</p>
                </div>
                <div className="mini-card mini-card-accent">
                  <span>Adjusted plan</span>
                  <strong>{(whatIfResult.scenario.probability_yes * 100).toFixed(1)}%</strong>
                  <p>{whatIfResult.scenario.risk_band}</p>
                </div>
                <div className="mini-card">
                  <span>Net uplift</span>
                  <strong>{whatIfResult.delta.probability_uplift_percent_points.toFixed(1)} pts</strong>
                  <p>{whatIfResult.delta.recommendation}</p>
                </div>
              </div>
              <div className="change-list">
                {whatIfResult.delta.changed_fields.map((item) => (
                  <div className="change-pill" key={item}>
                    {item}
                  </div>
                ))}
              </div>
            </div>
          )}
        </article>

        <article className="panel-card">
          <div className="panel-header">
            <h2>Model Drivers</h2>
            <p>Top variables influencing subscription propensity in the production model</p>
          </div>

          <div className="importance-list">
            {topFeatures.slice(0, 8).map((item) => (
              <div className="importance-row" key={item.feature}>
                <div className="importance-copy">
                  <strong>{normalizeLabel(item.feature)}</strong>
                  <span>Importance {item.importance.toFixed(4)}</span>
                </div>
                <div className="importance-bar">
                  <div
                    className="importance-bar-fill"
                    style={{ width: `${Math.max(item.importance * 100, 8)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="panel-card batch-panel">
        <div className="panel-header">
          <h2>Batch Scoring Workspace</h2>
          <p>Upload a campaign list, score the portfolio, and review the first recommended records</p>
        </div>

        <div className="batch-shell">
          <div>
            <label className="upload-zone">
              <input
                type="file"
                accept=".csv"
                onChange={(event) => setBatchFile(event.target.files?.[0] || null)}
              />
              <div>
                <strong>{batchFile ? batchFile.name : "Select a CSV file"}</strong>
                <p>Use the same scoring schema as the API input payload.</p>
              </div>
            </label>

            <button className="secondary-button" onClick={submitBatch} disabled={!batchFile || batchLoading}>
              {batchLoading ? "Processing..." : "Run batch scoring"}
            </button>
          </div>

          <div>
            <div className="batch-summary-grid">
              <div className="mini-card">
                <span>Rows scored</span>
                <strong>{formatNumber(batchResult?.summary.rows_scored)}</strong>
              </div>
              <div className="mini-card">
                <span>Priority leads</span>
                <strong>{formatNumber(batchResult?.summary.predicted_yes)}</strong>
              </div>
              <div className="mini-card">
                <span>Conversion rate</span>
                <strong>{formatPercent(batchResult?.summary.yes_rate_percent || 0)}</strong>
              </div>
              <div className="mini-card">
                <span>Avg score</span>
                <strong>{batchResult ? `${(Number(batchResult.summary.avg_probability_yes) * 100).toFixed(1)}%` : "0.0%"}</strong>
              </div>
            </div>

            {batchResult ? (
              <div className="table-wrap">
                <div className="table-toolbar">
                  <div className="table-toolbar-copy">
                    <strong>
                      Showing {formatNumber(batchStartIndex + 1)}-
                      {formatNumber(Math.min(batchStartIndex + batchPageSize, batchRecords.length))} of{" "}
                      {formatNumber(batchRecords.length)}
                    </strong>
                    <p>Browse scored customers page by page.</p>
                  </div>

                  <div className="table-toolbar-actions">
                    <label className="page-size-control">
                      <span>Rows</span>
                      <select
                        value={batchPageSize}
                        onChange={(event) => {
                          setBatchPageSize(Number(event.target.value));
                          setBatchPage(1);
                        }}
                      >
                        {[10, 20, 50, 100].map((size) => (
                          <option key={size} value={size}>
                            {size}
                          </option>
                        ))}
                      </select>
                    </label>

                    <div className="pagination-controls">
                      <button
                        type="button"
                        className="pagination-button"
                        onClick={() => setBatchPage((page) => Math.max(1, page - 1))}
                        disabled={currentBatchPage === 1}
                      >
                        Previous
                      </button>
                      <span className="pagination-status">
                        Page {formatNumber(currentBatchPage)} / {formatNumber(totalBatchPages)}
                      </span>
                      <button
                        type="button"
                        className="pagination-button"
                        onClick={() => setBatchPage((page) => Math.min(totalBatchPages, page + 1))}
                        disabled={currentBatchPage === totalBatchPages}
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>

                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Label</th>
                      <th>Probability yes</th>
                      <th>Risk band</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedBatchRecords.map((row, index) => (
                      <tr key={`${row.label}-${index}`}>
                        <td>{batchStartIndex + index + 1}</td>
                        <td>{row.label}</td>
                        <td>{(Number(row.probability_yes) * 100).toFixed(2)}%</td>
                        <td>{row.risk_band}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state empty-state-compact">
                <strong>No batch processed yet</strong>
                <p>Upload a portfolio file to review campaign-level opportunity and lead prioritization.</p>
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
