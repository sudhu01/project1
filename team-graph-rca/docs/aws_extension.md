# AWS extension preparation

The implemented `EC2StatusTemplate` prepares one read-only status call for one
registered EC2 instance. It does not connect to AWS. A deployment registry,
approved region list and measured AWS captures are required before adding an
executing backend. The Docker model and Q tables have no AWS accuracy claim.

The template uses `ec2 describe-instance-status`, an exact instance ID,
`--include-all-instances`, JSON output, disabled automatic pagination and disabled
interactive prompting. No additional arguments or endpoint overrides are accepted.
An empty response is missing data. A returned pagination token is rejected.
The normalizer checks instance identity and retains instance state and system and
instance health separately. A running instance is not proof of application health.

Each bundle declares one credit, a 10-second total deadline, a 3-second connection
timeout, a 5-second read timeout and at most 32 KiB of output. A future executor must
disable SDK/CLI retries, capture one subcommand receipt, enforce the output limit
while reading, and publish a timestamped `ToolResult`. It must not inspect a past
window using current state. Historical status needs a persisted sampler.

`argv()` returns structured arguments for direct execution. `bash()` quotes only
those validated arguments and checks the shell-token round trip. A policy supplies
a registered resource ID and never supplies a shell command. New ECS, CloudWatch
or Logs operations need their own bounded templates and normalized contracts.

References: [AWS status command](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-status.html)
and [AWS CLI pagination](https://docs.aws.amazon.com/cli/latest/userguide/cli-usage-pagination.html).
