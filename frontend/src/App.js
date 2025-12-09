import React, { useCallback, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE =
  process.env.REACT_APP_API_BASE?.replace(/\/$/, "") || "http://localhost:8000/api";
const PREDICT_URL = `${API_BASE}/predict/`;
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
    if (!result || typeof result.confidence !== "number") return null;
    return Math.round(result.confidence * 100);
  }, [result]);

  
  const isFood = result?.isFood;

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
        label: data.label,          
        isFood: data.is_food,
        foodGuess: data.guess,         
        confidence: data.confidence,  
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
              {typeof isFood === "boolean" && (
                <>
                  <span className={`badge ${isFood ? "food" : "not"}`}>
                    {isFood ? "Food" : "Non-Food"}
                  </span>

                  {isFood && result.foodGuess && (
                    <div style={{ marginTop: "0.5rem" }}>
                      Guess: <strong>{result.foodGuess}</strong>
                    </div>
                  )}
                </>
              )}

              {confidencePct !== null && (
                <div style={{ marginTop: "0.5rem" }}>
                  Confidence: <strong>{confidencePct}%</strong>
                </div>
              )}

              <details className="raw" style={{ marginTop: "0.5rem" }}>
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
