

class Bootstrap:
    # def __init__(self):
    #     pass

    def run():
        from src import pipeline
        pipeline.run_pipeline()


if __name__ == '__main__':
    app = Bootstrap
    app.run()