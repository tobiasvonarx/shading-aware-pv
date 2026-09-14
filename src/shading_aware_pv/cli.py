"""Start the solar workspace using shared .env and native runtime settings."""


def main():
    from building_data.runtime import launch

    launch("shading_aware_pv.api:create_app", default_port=5002)
