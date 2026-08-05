#!/usr/bin/env python3
"""Point a RunPod Serverless endpoint's template at a new Docker image.

Reads the endpoint's current template via the RunPod GraphQL API, then
re-saves that same template with only `imageName` changed - every other
field (GPU disk size, env vars, etc.) is carried over untouched so the rest
of the endpoint's config isn't clobbered.

Runs in dry-run mode by default (prints what would change, does nothing).
Pass --apply to actually call the saveTemplate mutation.

Usage:
    RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... \\
        python update_runpod_image.py <new-image-name:tag> [--apply]

Meant to be run by hand first (to confirm the query/mutation shape and that
nothing gets wiped) before it's wired into CI.
"""
import json
import os
import sys
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"


def _graphql(api_key: str, query: str, variables: dict = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare (fronting RunPod's API) returns a 403 (error 1010,
            # "browser fingerprint" block) for urllib's default User-Agent.
            "User-Agent": "runpod-endpoint-updater/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if "errors" in data:
        raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
    return data["data"]


def get_endpoint_template(api_key: str, endpoint_id: str) -> dict:
    """Return the custom template currently attached to `endpoint_id`.

    ``myself.podTemplates`` only lists RunPod's public/gallery templates, not
    custom ones - the endpoint's own template has to be read via the nested
    ``endpoints { template { ... } }`` field instead.
    """
    data = _graphql(
        api_key,
        """
        query myself {
            myself {
                endpoints {
                    id
                    template {
                        id
                        name
                        imageName
                        containerDiskInGb
                        volumeInGb
                        volumeMountPath
                        dockerArgs
                        isServerless
                        env { key value }
                    }
                }
            }
        }
        """,
    )
    endpoints = data["myself"]["endpoints"]
    endpoint = next((e for e in endpoints if e["id"] == endpoint_id), None)
    if endpoint is None:
        raise ValueError(f"No endpoint found with id={endpoint_id}")
    return endpoint["template"]


def save_template(api_key: str, template: dict, new_image_name: str) -> dict:
    """Re-save `template` with imageName replaced, everything else untouched."""
    env_input = [
        {"key": e["key"], "value": e["value"]} for e in template.get("env", [])
    ]
    variables = {
        "input": {
            "id": template["id"],
            "name": template["name"],
            "imageName": new_image_name,
            "containerDiskInGb": template["containerDiskInGb"],
            "volumeInGb": template["volumeInGb"],
            "volumeMountPath": template["volumeMountPath"],
            "dockerArgs": template["dockerArgs"],
            "isServerless": template["isServerless"],
            "env": env_input,
        }
    }
    data = _graphql(
        api_key,
        """
        mutation saveTemplate($input: SaveTemplateInput!) {
            saveTemplate(input: $input) {
                id
                imageName
            }
        }
        """,
        variables,
    )
    return data["saveTemplate"]


def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    positional = [a for a in args if a != "--apply"]
    if len(positional) != 1:
        print(__doc__)
        sys.exit(1)
    new_image_name = positional[0]

    api_key = os.environ["RUNPOD_API_KEY"]
    endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]

    template = get_endpoint_template(api_key, endpoint_id)
    print(f"Endpoint {endpoint_id} -> template {template['id']} ({template['name']})")
    print(f"  current imageName: {template['imageName']}")
    print(f"  new imageName:     {new_image_name}")
    print(f"  containerDiskInGb: {template['containerDiskInGb']}")
    print(f"  volumeInGb:        {template['volumeInGb']}")
    print(f"  env vars:          {[e['key'] for e in template.get('env', [])]}")

    if template["imageName"] == new_image_name:
        print("Image already up to date, nothing to do.")
        return

    if not apply:
        print("\nDry-run only (no --apply passed) - nothing changed.")
        return

    result = save_template(api_key, template, new_image_name)
    print(f"\nSaved. Template {result['id']} now points at {result['imageName']}")


if __name__ == "__main__":
    main()
