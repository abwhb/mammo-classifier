"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { Prediction } from "@/lib/types";

export function ResultCard({ prediction }: { prediction: Prediction }) {
  const isMalignant = prediction.label === "malignant";
  const pct = Math.round(prediction.confidence * 100);
  const probMalignantPct = Math.round(prediction.malignant_probability * 100);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-xl">Result</CardTitle>
            <CardDescription>
              Research / demonstration only — not for clinical decision-making
            </CardDescription>
          </div>
          <Badge
            variant={isMalignant ? "destructive" : "secondary"}
            className="text-sm uppercase tracking-wider"
          >
            {prediction.label}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        <div>
          <div className="flex justify-between mb-1.5 text-sm">
            <span className="text-zinc-600 dark:text-zinc-400">Confidence in label</span>
            <span className="font-mono font-medium">{pct}%</span>
          </div>
          <Progress value={pct} />
        </div>

        <div>
          <div className="flex justify-between mb-1.5 text-sm">
            <span className="text-zinc-600 dark:text-zinc-400">Probability of malignancy</span>
            <span className="font-mono font-medium">{probMalignantPct}%</span>
          </div>
          <Progress value={probMalignantPct} />
          <p className="text-xs text-zinc-500 mt-1">
            Decision threshold: {prediction.threshold.toFixed(3)}
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs pt-2 border-t border-zinc-200 dark:border-zinc-800">
          <dt className="text-zinc-500">Model version</dt>
          <dd className="font-mono text-right truncate">{prediction.model_version}</dd>
          <dt className="text-zinc-500">Latency</dt>
          <dd className="font-mono text-right">{prediction.latency_ms.toFixed(0)} ms</dd>
          <dt className="text-zinc-500">Request ID</dt>
          <dd className="font-mono text-right truncate" title={prediction.request_id}>
            {prediction.request_id.slice(0, 8)}…
          </dd>
        </dl>
      </CardContent>
    </Card>
  );
}
