import { UploadForm } from "@/components/upload-form";

export default function Home() {
  return (
    <main className="min-h-screen px-6 py-12 bg-zinc-50 dark:bg-zinc-950">
      <div className="max-w-2xl mx-auto space-y-8">
        <header className="text-center space-y-3">
          <p className="text-xs uppercase tracking-widest text-zinc-500">
            Research / demonstration only — not for clinical use
          </p>
          <h1 className="text-3xl font-semibold tracking-tight">Mammogram Classifier</h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 max-w-md mx-auto">
            Upload a screening mammogram (DICOM, PNG, or JPEG). Uploaded images are
            de-identified server-side and held in memory only — not persisted.
          </p>
        </header>

        <UploadForm />

        <footer className="text-center text-xs text-zinc-500 pt-6 border-t border-zinc-200 dark:border-zinc-800">
          Built for the FAHM Biotechnology technical assessment.
          <br />
          Not a medical device. Not approved or intended for clinical use.
        </footer>
      </div>
    </main>
  );
}
