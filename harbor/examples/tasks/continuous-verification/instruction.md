Find the six-character lowercase password. It is not written down anywhere you can read.

Submit a guess by writing it to a file in `/app/submissions/queue/`, for example:

```bash
echo -n "abcdef" > /app/submissions/queue/try-1.txt
```

A file whose name starts with `.` is ignored, so write there and rename into
place if you want to be certain a large file is never read half-written.

That returns as soon as the write does. Some seconds later a result appears at
`/app/submissions/results/try-1.txt.json`, containing how many characters of
your guess were correct and in the right position. Keep guessing, using each
result to narrow down the next one. Poll for the file; do not block waiting on it.

When you are confident, write the answer to `/app/submission/answer.txt`. That
file is what gets graded at the end.
