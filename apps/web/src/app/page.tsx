import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center px-6 bg-zinc-50 dark:bg-zinc-950">
      <div className="max-w-xl text-center space-y-6">
        <p className="text-xs uppercase tracking-widest text-zinc-500">
          Research / demo only — not for clinical use
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Mammogram Classifier
        </h1>
        <p className="text-zinc-600 dark:text-zinc-400">
          Upload UI is shipping in Phase 4. The API proxy is live at{" "}
          <Link href="/api/predict" className="underline font-mono">
            /api/predict
          </Link>
          .
        </p>
      </div>
    </main>
  );
}
