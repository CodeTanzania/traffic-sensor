#ifndef TRAFFIC_SENSOR_DB_HANDLER_H
#define TRAFFIC_SENSOR_DB_HANDLER_H

#include <fstream>
#include <string>
#include <sqlite3.h>

namespace com
{
    namespace github
    {
        namespace codetanzania
        {
            class DbHandler {
            public:
                DbHandler(const std::string &config_file);
            private:
                std::string config_file;
                std::string database_name;
                void loadDbConfig();
            };

            DbHandler::DbHandler(const std::string &config_file)
            {
                std::ifstream f(config_file, std::ifstream::binary);

            }
        }
    }
}

#endif // TRAFFIC_SENSOR_DB_HANDLER_H
