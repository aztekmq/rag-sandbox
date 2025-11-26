# Verifying Docker Volume Mapping for This Repository

The commands below help confirm that the repository on the host is correctly mounted into a running Docker container. Each command favors verbose or inspect-style output so you can debug mapping problems quickly.

## 1) Confirm the container and mount path

```bash
# List running containers and the command used to launch them
# The `--no-trunc` flag shows the full command line for better visibility.
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Mounts}}\t{{.Command}}' --no-trunc

# Inspect the container to see the bind mount target and source
# Replace <container_name_or_id> with the actual container identifier.
docker inspect --type container <container_name_or_id> \
  --format 'Name: {{.Name}}\nMounts: {{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

## 2) Check that the repo path exists inside the container

```bash
# Enter the container with a shell. Use the same user that owns the files if possible.
docker exec -it <container_name_or_id> /bin/bash

# Once inside the container, verify the expected mount point is present.
set -euxo pipefail
pwd
ls -al /workspace/rag-sandbox
```

## 3) Validate bidirectional write/read behavior

```bash
# From inside the container, create a marker file in the repo mount.
set -euxo pipefail
echo "volume-check-$(date -Iseconds)" > /workspace/rag-sandbox/.docker-volume-marker
cat /workspace/rag-sandbox/.docker-volume-marker

# Exit the container and confirm the file is visible on the host.
exit
ls -al /workspace/rag-sandbox/.docker-volume-marker
cat /workspace/rag-sandbox/.docker-volume-marker
```

## 4) Test host-to-container sync

```bash
# On the host, append a line to the marker file.
echo "host-update-$(date -Iseconds)" >> /workspace/rag-sandbox/.docker-volume-marker
cat /workspace/rag-sandbox/.docker-volume-marker

# Verify the change from inside the container.
docker exec -it <container_name_or_id> /bin/bash -lc "set -euxo pipefail; cat /workspace/rag-sandbox/.docker-volume-marker"
```

## 5) Optional: verify permissions and UID/GID alignment

```bash
# Check ownership and permissions of the mounted files from inside the container.
docker exec -it <container_name_or_id> /bin/bash -lc "set -euxo pipefail; stat -c '%n %U:%G %a' /workspace/rag-sandbox"

# Compare with the host permissions to ensure consistency.
stat -c '%n %U:%G %a' /workspace/rag-sandbox
```

## Cleanup

```bash
# Remove the marker file from host and container after testing.
rm -f /workspace/rag-sandbox/.docker-volume-marker
```

These steps provide verbose, inspectable output at each stage so you can quickly diagnose whether the host repository is correctly mapped into the Docker container and that changes propagate in both directions.
