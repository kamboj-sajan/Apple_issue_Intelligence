"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useApi } from "@/lib/api";

const NAV = [
  ["/", "Overview"],
  ["/issues", "Emerging issues"],
  ["/graph", "Knowledge graph"],
  ["/related", "Related questions"],
  ["/drift", "Drift monitoring"],
  ["/models", "Model evaluation"],
];

function DataSourceBanner() {
  const { data, error } = useApi("/health");
  if (error) {
    return (
      <div role="status" className="border-b border-critical/40 bg-critical/10 px-6 py-2 text-sm">
        <strong className="text-critical">API unreachable.</strong> Start the backend (<code>uvicorn backend.app.main:app</code>) — {error.message}
      </div>
    );
  }
  const src = data?.meta?.data_source;
  if (src === "demo") {
    return (
      <div role="status" className="border-b border-warning/50 bg-warning/20 px-6 py-2 text-sm text-ink">
        <strong>Demo data.</strong> Everything shown is a small synthetic dataset, not real Ask Different questions.
        Load real results with <code>aii-db load</code>.
      </div>
    );
  }
  if (src === "empty") {
    return (
      <div role="status" className="border-b border-line bg-surface-2 px-6 py-2 text-sm">
        <strong>No data loaded.</strong> Run the pipeline and <code>aii-db load</code> (or <code>aii-db load-demo</code>).
      </div>
    );
  }
  if (src === "real") {
    return (
      <div role="status" className="border-b border-line bg-surface px-6 py-2 text-xs text-ink-2">
        <span className="mr-2 inline-flex items-center gap-1 font-medium text-good">● Real data</span>
        Calculated from public Ask Different questions · snapshot <code>{data.meta.snapshot_id}</code>. Results are
        candidate discussion patterns, not confirmed defects.
      </div>
    );
  }
  return null;
}

export default function Shell({ children }) {
  const path = usePathname();
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-56 shrink-0 border-r border-line bg-surface px-4 py-6 md:block">
        <div className="mb-8 px-2">
          <div className="text-sm font-semibold">Apple Issue Intelligence</div>
          <div className="text-xs text-muted">Ask Different research dashboard</div>
        </div>
        <nav className="flex flex-col gap-1">
          {NAV.map(([href, label]) => {
            const active = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link key={href} href={href}
                className={`rounded-md px-2 py-1.5 text-sm ${active ? "bg-accent/15 font-medium text-accent" : "text-ink-2 hover:bg-surface-2"}`}>
                {label}
              </Link>
            );
          })}
        </nav>
        <p className="mt-10 px-2 text-xs leading-relaxed text-muted">
          Content from Ask Different (CC BY-SA), via the Stack Exchange API. Each question links to its source.
        </p>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <DataSourceBanner />
        <nav className="flex gap-3 overflow-x-auto border-b border-line bg-surface px-4 py-2 text-sm md:hidden">
          {NAV.map(([href, label]) => (
            <Link key={href} href={href} className="whitespace-nowrap text-ink-2">{label}</Link>
          ))}
        </nav>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 md:px-8">{children}</main>
      </div>
    </div>
  );
}
