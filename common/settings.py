"""Single source of env config. Nothing else in the codebase reads os.environ."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    TFY_TOKEN: str = ""
    TFY_GATEWAY_URL: str = "https://gateway.truefoundry.ai"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""
    OSTEON_TRACE_DIR: str = "./traces"
    OSTEON_DEADLINE_MS: int = 20000


settings = Settings()
