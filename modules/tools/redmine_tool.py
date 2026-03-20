from __future__ import annotations

from services.integrations import REDMINE_SERVICE_NAME, RedmineConfigurationError


def build_get_issue_handler(kernel):
    def get_issue(issue_id):
        service = kernel.get_service(REDMINE_SERVICE_NAME)
        if service is None:
            raise RedmineConfigurationError("Redmine service is not registered")
        return service.client.get_issue(issue_id)

    return get_issue
