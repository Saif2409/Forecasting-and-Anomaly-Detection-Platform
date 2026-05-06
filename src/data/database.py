from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import settings


def get_engine(database_url: Optional[str] = None) -> Engine:
    return create_engine(database_url or settings.database_url, pool_pre_ping=True)


def is_database_available(database_url: Optional[str] = None) -> bool:
    try:
        engine = get_engine(database_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def write_dataframe_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    database_url: Optional[str] = None,
    if_exists: str = "replace",
) -> bool:
    try:
        engine = get_engine(database_url)
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        return True
    except SQLAlchemyError as exc:
        print(f"PostgreSQL load skipped: {exc}")
        return False


def read_table_from_postgres(table_name: str, database_url: Optional[str] = None) -> pd.DataFrame:
    engine = get_engine(database_url)
    return pd.read_sql_table(table_name, engine)
