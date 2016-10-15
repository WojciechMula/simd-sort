#include <cstring>
#include <string>
#include <unordered_set>

class CommandLine final {
    int argc;
    char** argv;

    std::unordered_set<std::string> arguments;

public:
    CommandLine(int argc, char* argv[])
        : argc(argc)
        , argv(argv) {

        for (int i=1; i < argc; i++) {
            arguments.insert(argv[i]);
        }
    }

    bool has(const std::string& name) const {
        return arguments.count(name);
    }

    bool empty() const {
        return arguments.empty();
    }
};
