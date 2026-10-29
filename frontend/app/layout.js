import "./globals.css";
import Shell from "@/components/Shell";

export const metadata = {
  title: "Apple Issue Intelligence",
  description: "Emerging-issue signals, related questions, knowledge graph and drift monitoring for Ask Different.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
