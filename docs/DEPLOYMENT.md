# Deployment and local runtime

Pathagon is built with Vinext/Vite and deployed through Sites. The deployable
artifact is produced by the pushed source commit; local `dist/` output is
disposable.

## Local development

```bash
npm run dev
npm run build
npm run start
```

`npm run build` creates the application artifact and packages the checked-in
Drizzle migrations and Sites configuration. The generated `.sites-runtime/`,
`.wrangler/`, and `dist/` directories are disposable and ignored.

## Database

`.openai/hosting.json` declares the D1 binding. Schema changes are made through
Drizzle migrations in `drizzle/`; request handlers never create tables at
request time.

Apply migrations before enabling archive writes in a deployment.

## Sign in with ChatGPT

`app/chatgpt-auth.ts` provides optional helpers for reading the current user or
requiring sign-in. Dispatch owns the reserved sign-in, callback, cookie, and
identity-header routes. The application should not implement those routes.

Sign-in establishes identity, not workspace membership. Use the Sites access
policy or an explicit server-side allowlist when a route needs restricted
access.

## Release checklist

1. Run the relevant JavaScript, Rust, Python, and parity tests.
2. Confirm generated model/WASM hashes and the associated evaluation report.
3. Commit a coherent source and artifact milestone.
4. Push the commit and deploy from that commit.
5. Verify the public build and archive read paths.
