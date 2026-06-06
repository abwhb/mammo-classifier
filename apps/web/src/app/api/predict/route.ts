import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const API_URL = process.env.API_URL ?? "http://127.0.0.1:8080";
const INTERNAL_TOKEN = process.env.INTERNAL_TOKEN;
const MAX_BYTES = 20 * 1024 * 1024;

export async function POST(req: NextRequest) {
  const incoming = await req.formData();
  const file = incoming.get("file");
  if (!(file instanceof File)) {
    return NextResponse.json({ error: "missing file" }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return NextResponse.json({ error: "file too large" }, { status: 413 });
  }

  const upstreamForm = new FormData();
  upstreamForm.append("file", file, file.name);

  const headers: Record<string, string> = {};
  if (INTERNAL_TOKEN) headers["x-internal-token"] = INTERNAL_TOKEN;

  let upstream: Response;
  try {
    upstream = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: upstreamForm,
      headers,
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ error: "inference service unreachable" }, { status: 502 });
  }

  const payload = await upstream.json().catch(() => ({ error: "invalid upstream response" }));
  return NextResponse.json(payload, { status: upstream.status });
}
