# Dashboard

Next.js 15 (App Router, JavaScript), Tailwind CSS 4, Recharts and React Flow. It reads everything from
the FastAPI backend; set `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

```bash
npm ci
npm run dev      # http://localhost:3000
npm run build    # production build (standalone output, used by the Dockerfile)
npx next lint
```

Every page shows whether the API is serving **real** results or the **synthetic demo** dataset.
Chart colours come from a colour-vision-deficiency-validated palette defined as CSS tokens in
`app/globals.css`, with separately chosen dark-mode steps.
