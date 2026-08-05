import "./globals.css";

export const metadata = {
  title: "R26-IT-143 — Smart Pool Monitoring",
  description: "AI-Based Smart Swimming Pool Monitoring System",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
