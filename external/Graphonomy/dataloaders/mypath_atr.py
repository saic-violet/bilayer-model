class Path(object):
    @staticmethod
    def db_root_dir(database):
        """
        Return the root directory of a database.

        Args:
            database: (str): write your description
        """
        if database == 'atr':
            return './data/datasets/ATR/'  # folder that contains atr/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
