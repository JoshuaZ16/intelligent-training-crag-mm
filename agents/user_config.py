"""AIcrowd entrypoint for the verified week-2 agent.

Set CRAG_TASK_MODE to vision, task1, or task2.  The repository's current
aicrowd.json targets single-source augmentation, so task1 is the default.
"""

from agents.course_agent_v2 import CourseRAGAgentV2

UserAgent = CourseRAGAgentV2
