"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { absoluteUrl, checkFile, createJob, getJob, type Job } from "@/lib/api";
import styles from "./Studio.module.css";

const CharacterViewer = dynamic(() => import("./CharacterViewer"), {
  ssr: false,
  loading: () => <div className={styles.placeholder}>Loading viewer…</div>,
});

const STAGE_LABELS: Record<string, string> = {
  validating_input: "Checking image",
  removing_background: "Isolating the person",
  reconstructing: "Reconstructing 3D shape",
  processing_mesh: "Cleaning up the mesh",
  texturing: "Baking the texture",
  exporting: "Exporting GLB",
};

export default function Studio() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const busy = submitting || job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    if (!preview) return;
    return () => URL.revokeObjectURL(preview);
  }, [preview]);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed") return;
    const timer = setTimeout(async () => {
      try {
        setJob(await getJob(job.id));
      } catch (e) {
        setError((e as Error).message);
        setJob(null);
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [job]);

  const choose = useCallback((f: File | undefined) => {
    if (!f) return;
    const problem = checkFile(f);
    setError(problem);
    if (problem) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setJob(null);
  }, []);

  const generate = async () => {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    try {
      setJob(await createJob(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const result = job?.status === "succeeded" ? job.result : undefined;
  const modelUrl = result ? absoluteUrl(result.glb_url) : null;

  return (
    <div className={styles.studio}>
      <section className={styles.panel}>
        <h2>1. Photo</h2>
        <div
          className={`${styles.drop} ${dragging ? styles.dragging : ""}`}
          onClick={() => !busy && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (!busy) choose(e.dataTransfer.files[0]);
          }}
          role="button"
          tabIndex={0}
        >
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={preview} alt="Selected photo" className={styles.preview} />
          ) : (
            <p>
              Drop a PNG or JPG here, or click to choose.
              <br />
              <small>Best results: one person, front-facing, plain background, good light.</small>
            </p>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          hidden
          onChange={(e) => {
            choose(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <button className={styles.primary} disabled={!file || busy} onClick={generate}>
          {busy ? "Generating…" : "Generate 3D character"}
        </button>

        {job && job.status !== "succeeded" && job.status !== "failed" && (
          <div className={styles.progress}>
            <div className={styles.bar}>
              <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
            </div>
            <p>
              {job.status === "queued"
                ? `Queued${job.queue_position ? ` (${job.queue_position} ahead)` : ""}…`
                : `${STAGE_LABELS[job.stage ?? ""] ?? "Working"}… ${Math.round(job.progress * 100)}%`}
            </p>
            <small>Runs on CPU; expect about 1–2 minutes. The first run also downloads model weights.</small>
          </div>
        )}
        {job?.status === "failed" && <p className={styles.error}>Generation failed: {job.error}</p>}
        {error && <p className={styles.error}>{error}</p>}
      </section>

      <section className={`${styles.panel} ${styles.viewerPanel}`}>
        <h2>2. Character</h2>
        <div className={styles.viewer}>
          {modelUrl ? (
            <CharacterViewer url={modelUrl} />
          ) : (
            <div className={styles.placeholder}>Your 3D character will appear here. Drag to rotate, scroll to zoom.</div>
          )}
        </div>
        {result && modelUrl && (
          <div className={styles.result}>
            <a className={styles.primary} href={modelUrl} download>
              Download .glb
            </a>
            <p className={styles.meta}>
              {result.reconstructor} · {result.stats.triangles?.toLocaleString()} triangles ·{" "}
              {(result.stats.bytes / 1024 / 1024).toFixed(1)} MB ·{" "}
              {Object.values(result.timings)
                .reduce((a, b) => a + b, 0)
                .toFixed(0)}
              s
            </p>
            {result.warnings.map((w) => (
              <p key={w} className={styles.warning}>
                {w}
              </p>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
