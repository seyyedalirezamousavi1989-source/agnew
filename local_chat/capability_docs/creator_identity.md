# Creator identity

## Use

Use the local identity flow to verify and persist the configured creator profile. Questions about the current user or the agent's creator are answered from the local database without sending identity fields to Ollama.

## Inputs and outputs

- Input: a six-digit local verification code, or SMTP settings when email verification is configured; creator-identification questions.
- Output: verification status, persisted creator profile, or a program-generated identity answer.

## Dependencies and limitations

- Uses SQLite on the local device.
- A locally configured code is convenient for local use, but does not prove email ownership and can be read by anyone who can read the settings file.
- Email verification requires valid SMTP credentials and network access.
- The profile is local to this installation and is not user/role/permission management.
