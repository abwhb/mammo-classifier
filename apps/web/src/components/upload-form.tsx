"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ResultCard } from "@/components/result-card";
import { isPrediction, type Prediction } from "@/lib/types";

type Status = "idle" | "uploading" | "done" | "error";

const ACCEPT = {
  "application/dicom": [".dcm"],
  "image/png": [".png"],
  "image/jpeg": [".jpg", ".jpeg"],
};
const MAX_BYTES = 20 * 1024 * 1024;

export function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    setError(null);
    setPrediction(null);
    setStatus("idle");
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const isImage = f.type.startsWith("image/");
    setPreviewUrl(isImage ? URL.createObjectURL(f) : null);
  }, [previewUrl]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxFiles: 1,
    maxSize: MAX_BYTES,
    multiple: false,
  });

  async function submit() {
    if (!file) return;
    setStatus("uploading");
    setError(null);
    setPrediction(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const json = await res.json();
      if (!res.ok) {
        setStatus("error");
        setError(typeof json?.error === "string" ? json.error : `request failed (${res.status})`);
        return;
      }
      if (!isPrediction(json)) {
        setStatus("error");
        setError("unexpected response shape from server");
        return;
      }
      setPrediction(json);
      setStatus("done");
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "network error");
    }
  }

  function reset() {
    setFile(null);
    setPrediction(null);
    setError(null);
    setStatus("idle");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  }

  return (
    <div className="space-y-6">
      <div
        {...getRootProps()}
        className={[
          "border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors",
          "bg-white dark:bg-zinc-900",
          isDragActive
            ? "border-zinc-900 dark:border-zinc-100 bg-zinc-50 dark:bg-zinc-800"
            : "border-zinc-300 dark:border-zinc-700 hover:border-zinc-500",
        ].join(" ")}
      >
        <input {...getInputProps()} />
        <div className="space-y-2">
          <p className="text-sm font-medium">
            {isDragActive ? "Drop the file here" : "Drag a mammogram here, or click to browse"}
          </p>
          <p className="text-xs text-zinc-500">
            DICOM (.dcm), PNG, or JPEG · max 20&nbsp;MB
          </p>
        </div>
      </div>

      {file && (
        <div className="flex items-center justify-between rounded-lg border border-zinc-200 dark:border-zinc-800 px-4 py-3 bg-zinc-50 dark:bg-zinc-900/50">
          <div className="min-w-0 flex-1 space-y-0.5">
            <p className="text-sm font-medium truncate" title={file.name}>
              {file.name}
            </p>
            <p className="text-xs text-zinc-500">
              {(file.size / 1024).toFixed(0)}&nbsp;KB · {file.type || "application/dicom"}
            </p>
          </div>
          <div className="flex gap-2 ml-3">
            <Button
              size="sm"
              variant="outline"
              onClick={reset}
              disabled={status === "uploading"}
            >
              Reset
            </Button>
            <Button size="sm" onClick={submit} disabled={status === "uploading"}>
              {status === "uploading" ? "Analyzing…" : "Analyze"}
            </Button>
          </div>
        </div>
      )}

      {previewUrl && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={previewUrl}
          alt="Preview of uploaded mammogram"
          className="rounded-lg max-h-72 mx-auto object-contain bg-black"
        />
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Inference failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {prediction && <ResultCard prediction={prediction} />}
    </div>
  );
}
