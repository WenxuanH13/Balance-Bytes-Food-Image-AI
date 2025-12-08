import React, { useCallback, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE =
  process.env.REACT_APP_API_BASE?.replace(/\/$/, "") || "http://localhost:8000/api";
const PREDICT_URL = `${API_BASE}/predict`;
const IMAGE_FIELD = "image"; // change if backend expects a different field

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const inputRef = useRef(null);

  const pickFile = () => inputRef.current?.click();

  const onFileSelected = useCallback((f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file (jpg, png, webp, etc.)");
      setFile(null);
      setPreview("");
      return;
    }
    setError(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const handleInputChange = (e) => onFileSelected(e.target.files?.[0] || null);

  // Drag & drop
  const [dragOver, setDragOver] = useState(false);
  const onDragOver  = (e) => { e.preventDefault(); setDragOver(true);  };
  const onDragLeave = (e) => { e.preventDefault(); setDragOver(false); };
  const onDrop      = (e) => { e.preventDefault(); setDragOver(false); onFileSelected(e.dataTransfer.files?.[0] || null); };

  const confidencePct = useMemo(() => {
    if (!result) return null;
    const c = result.confidence ?? result.raw?.probability ?? result.raw?.score;
    return typeof c === "number" ? Math.round(c * 100) : null;
  }, [result]);

  const isFood = useMemo(() => {
    if (!result?.label) return null;
    const l = String(result.label).toLowerCase();
    return l.includes("food");
  }, [result]);

  const reset = () => { setFile(null); setPreview(""); setResult(null); setError(null); };

  const submit = async () => {
    if (!file) { setError("Please select an image first."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData();
      fd.append(IMAGE_FIELD, file);

      const res = await fetch(PREDICT_URL, { method: "POST", body: fd });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Request failed (${res.status}): ${text}`);
      }
      const data = await res.json();
      setResult({
        label: data.label ?? data.prediction ?? data.class ?? undefined,
        confidence:
          typeof data.confidence === "number"
            ? data.confidence
            : typeof data.probability === "number"
            ? data.probability
            : typeof data.score === "number"
            ? data.score
            : undefined,
        raw: data,
      });
    } catch (err) {
      console.error(err);
      setError(err?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="wrap">
        <h1 className="h1">Food / Non-Food Detector</h1>
        <p className="subtitle">
          Upload an image and we’ll call our model to predict the label with confidence.
        </p>

        <div className="card">
          <div
            className={`dropzone ${dragOver ? "drag" : ""}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            {preview ? (
              <img alt="preview" src={preview} className="preview" />
            ) : (
              <span>Drag & drop an image here</span>
            )}
          </div>

          <div className="actions">
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleInputChange}
            />
            <button className="btn black" onClick={pickFile} disabled={loading}>
              Choose Image
            </button>
            <button className="btn blue" onClick={submit} disabled={loading || !file}>
              {loading ? "Predicting…" : "Predict"}
            </button>
            {file && (
              <button className="btn gray" onClick={reset} disabled={loading}>
                Clear
              </button>
            )}
          </div>

          {error && (
            <div className="error">
              <strong>Error:</strong> {error}
            </div>
          )}

          {result && (
            <div className="meta">
              {isFood !== null && (
                <span className={`badge ${isFood ? "food" : "not"}`}>
                  {result.label || (isFood ? "Food" : "Non-Food")}
                </span>
              )}
              {confidencePct !== null && (
                <span>Confidence: <strong>{confidencePct}%</strong></span>
              )}
              <details className="raw">
                <summary>Raw response</summary>
                <pre>{JSON.stringify(result.raw ?? {}, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
