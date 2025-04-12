from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool

file_writer_tool_summary = FileWriterTool(filename='summary.txt', directory='meeting_minutes')
file_writer_tool_action_items = FileWriterTool(filename='action_items.txt', directory='meeting_minutes')
file_writer_tool_sentiment = FileWriterTool(filename='sentiment.txt', directory='meeting_minutes')

@CrewBase
class MeetingMinutesCrew():
    """MeetingMinutes Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def meeting_minutes_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minutes_summarizer"],
            tools=[
                file_writer_tool_summary,
                file_writer_tool_action_items,
                file_writer_tool_sentiment
            ]
        )

    @agent
    def meeting_minutes_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minutes_writer"],
        )

    @task
    def meeting_minutes_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minutes_summary_task"],
        )

    @task
    def meeting_minutes_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minutes_writing_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
