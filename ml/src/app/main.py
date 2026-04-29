

class Bootstrap:
    @staticmethod
    def run() -> None:
        from src.app import pipeline

        pipeline.run_training()


if __name__ == "__main__":
    Bootstrap.run()