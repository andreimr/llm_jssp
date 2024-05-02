from langchain.skills import Skill

class WeatherSkill(Skill):
    def __init__(self, weather_api):
        self.weather_api = weather_api

    def get_weather(self, location):
        # Implement the logic to fetch weather using the provided API
        weather_data = self.weather_api.fetch_weather(location)
        return weather_data