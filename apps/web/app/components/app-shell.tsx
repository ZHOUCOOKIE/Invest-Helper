"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

type ThemeMode = "light" | "dark";

type NavItem = {
  label: string;
  href: string;
  match: (pathname: string) => boolean;
};

function applyTheme(mode: ThemeMode): void {
  document.documentElement.dataset.theme = mode;
  window.localStorage.setItem("ip-theme", mode);
}

export function AppShell({
  children,
  todayDigestDate,
}: {
  children: React.ReactNode;
  todayDigestDate: string;
}) {
  const pathname = usePathname();
  const [glow, setGlow] = useState({ x: 0, y: 0, visible: false });

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      setGlow({ x: event.clientX, y: event.clientY, visible: true });
    };

    const onLeave = () => {
      setGlow((prev) => ({ ...prev, visible: false }));
    };

    const onDown = (event: PointerEvent) => {
      const wave = document.createElement("span");
      wave.className = "click-wave";
      wave.style.left = `${event.clientX}px`;
      wave.style.top = `${event.clientY}px`;
      document.body.appendChild(wave);
      window.setTimeout(() => {
        wave.remove();
      }, 700);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerleave", onLeave);
    window.addEventListener("pointerdown", onDown);

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", onLeave);
      window.removeEventListener("pointerdown", onDown);
    };
  }, []);

  const navItems = useMemo<NavItem[]>(
    () => [
      { label: "看板", href: "/dashboard", match: (path) => path.startsWith("/dashboard") },
      { label: "导入", href: "/ingest", match: (path) => path.startsWith("/ingest") },
      { label: "抽取审核", href: "/extractions", match: (path) => path.startsWith("/extractions") },
      { label: "资产", href: "/assets", match: (path) => path.startsWith("/assets") },
      { label: "KOL", href: "/kols", match: (path) => path.startsWith("/kols") },
      { label: "日报", href: `/digests/${todayDigestDate}`, match: (path) => path.startsWith("/digests") },
      { label: "健康检查", href: "/health", match: (path) => path.startsWith("/health") },
    ],
    [todayDigestDate],
  );

  return (
    <div className="app-shell">
      <div
        className={`mouse-glow ${glow.visible ? "visible" : ""}`}
        style={{ transform: `translate(${glow.x - 160}px, ${glow.y - 160}px)` }}
        aria-hidden="true"
      />
      <header className="top-tabs" aria-label="主导航">
        <div className="brand">InvestPulse</div>
        <nav className="top-tabs-nav">
          {navItems.map((item) => {
            const active = item.match(pathname);
            return (
              <Link key={item.label} href={item.href} className={`top-tab ${active ? "active" : ""}`}>
                {item.label}
              </Link>
            );
          })}
        </nav>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => {
            const current: ThemeMode = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
            const next = current === "light" ? "dark" : "light";
            applyTheme(next);
          }}
          aria-label="切换主题"
        >
          主题
        </button>
      </header>
      <div className="shell-content">{children}</div>
    </div>
  );
}
