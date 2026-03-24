import type { HTMLAttributes } from 'react'

import { cn } from '@/lib/utils'

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full bg-[var(--soft)] px-3 py-1 text-xs font-medium text-[var(--muted-ink)] ring-1 ring-black/6',
        className,
      )}
      {...props}
    />
  )
}
