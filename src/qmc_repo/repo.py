import intake
import s3fs


class Repo:
    def __init__(self):
        self.endpoint_url = "https://uri.osn.mghpcc.org"

        self.catalog = intake.open_catalog(
            's3://phy240060/configurations/configurations.yaml',
            storage_options={
                'anon': True,
                'endpoint_url': self.endpoint_url}
            )

        # Initialize s3fs to allow access to hdf5 files in S3
        self.fs = s3fs.S3FileSystem(
            endpoint_url=self.endpoint_url,
            anon=True
        )

    @property
    def entries(self):
        return [c for c in self.catalog]

    def get_source(self, name):
        return getattr(self.catalog, name)

    def __getattr__(self, name):
        # This is only called if the attribute isn't found on Repo itself
        return getattr(self.catalog, name)
