# Security Policy

## Supported Versions

Currently, only the latest release (`main` branch / `v1.x`) receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| v1.0.x  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Because this engine operates at the bare-metal level using direct `mmap` ingestion, raw pointer arithmetic, and lock-free concurrency, memory safety is paramount. Vulnerabilities like buffer overflows, out-of-bounds reads, or race conditions can lead to host-level exploitation.

**DO NOT OPEN A PUBLIC GITHUB ISSUE FOR SECURITY VULNERABILITIES.**

If you discover a security vulnerability within this project, please send an e-mail to the maintainers (or reach out via direct message if an email is not provided in the maintainer's GitHub profile).

### What to Include in Your Report
Please provide a detailed report including:
- A description of the vulnerability and its impact.
- The OS and GPU architecture you tested on.
- A Minimal Reproducible Example (MRE) or a proof-of-concept exploit (e.g., a maliciously crafted `.safetensors` file).
- Steps to reproduce.

### Response Time
We treat security reports with the highest priority and will acknowledge receipt of your vulnerability report within 48 hours, providing an estimated timeline for a patch.
