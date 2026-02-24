import { redirect } from "next/navigation";

export default function DigestEntryPage() {
  const today = new Date().toISOString().slice(0, 10);
  redirect(`/digests/${today}`);
}
