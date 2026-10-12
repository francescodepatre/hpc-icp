#include "io.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

    struct Prop{
        std::string name;
        int size = 0;
        char kind = 'f';
        bool is_list = false;
    };
    
    struct Element{
        std::string name;
        std::size_t count = 0;
        std::vector<Prop> props;
    };

    bool parse_type(const std::string& t, int& size, char& kind) {
        if (t == "float" || t == "float32") { 
            size = 4; 
            kind = 'f'; 
            return true; 
        }
        if (t == "double" || t == "float64") { 
            size = 8; 
            kind = 'f'; 
            return true; 
        }
        if (t == "char" || t == "int8") { 
            size = 1; 
            kind = 'i'; 
            return true; 
        }
        if (t == "uchar" || t == "uint8") { 
            size = 1; 
            kind = 'u'; 
            return true; 
        }
        if (t == "short" || t == "int16") { 
            size = 2; 
            kind = 'i'; 
            return true; 
        }
        if (t == "ushort" || t == "uint16") { 
            size = 2; 
            kind = 'u'; 
            return true; 
        }
        if (t == "int" || t == "int32") { 
            size = 4; 
            kind = 'i'; 
            return true; 
        }
        if (t == "uint" || t == "uint32") { 
            size = 4; 
            kind = 'u'; 
            return true; 
        }
        return false;
    }

    double read_scalar(const unsigned char* p, int size, char kind) {
        if (kind == 'f') {
            if (size == 4) { 
                float v; std::memcpy(&v, p, 4); 
                return v; 
            }
            double v; std::memcpy(&v, p, 8); 
            return v;
        }
        if (kind == 'i') {
            if (size == 1) { 
                int8_t v; 
                std::memcpy(&v, p, 1); 
                return v; }
            if (size == 2) { 
                int16_t v; 
                std::memcpy(&v, p, 2); 
                return v; }
            int32_t v; 
            std::memcpy(&v, p, 4); 
            return v;
        }
        if (size == 1) { 
            uint8_t v; 
            std::memcpy(&v, p, 1); 
            return v; }
        if (size == 2) { 
            uint16_t v; 
            std::memcpy(&v, p, 2); 
            return v; }
        uint32_t v; 
        std::memcpy(&v, p, 4); 
        return v;
    }

    void strip_cr(std::string& s) {
        if (!s.empty() && s.back() == '\r') 
            s.pop_back();
    }
    
}


bool load_ply(const std::string& path, PointCloud& pc, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = path + ": " + m;
        return false;
    };

    std::ifstream f(path, std::ios::binary);
    if (!f) 
        return fail("impossibile aprire il file");

    std::string line;
    std::getline(f, line);
    strip_cr(line);
    
    if (line != "ply") 
        return fail("non e' un file PLY");

    int fmt = -1;  // 0 = ascii, 1 = binary little endian
    std::vector<Element> elems;
    bool header_end = false;

    while (std::getline(f, line)) {
        strip_cr(line);
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        if (kw == "format") {
            std::string v;
            ss >> v;
            if (v == "ascii") fmt = 0;
            else if (v == "binary_little_endian") fmt = 1;
            else return fail("formato non supportato: " + v);
        } else if (kw == "element") {
            Element e;
            ss >> e.name >> e.count;
            elems.push_back(e);
        } else if (kw == "property") {
            if (elems.empty()) return fail("property prima di element");
            Prop p;
            std::string t;
            ss >> t;
            if (t == "list") {
                std::string ct, it;
                ss >> ct >> it >> p.name;
                p.is_list = true;
            } else {
                if (!parse_type(t, p.size, p.kind)) return fail("tipo sconosciuto: " + t);
                ss >> p.name;
            }
            elems.back().props.push_back(p);
        } else if (kw == "end_header") {
            header_end = true;
            break;
        }
    }
    if (!header_end || fmt < 0) 
        return fail("header PLY incompleto");

    pc.clear();

    for (const Element& e : elems) {
        bool has_list = false;
        size_t row = 0;
        for (const Prop& p : e.props) {
            if (p.is_list) has_list = true;
            else row += p.size;
        }

        if (e.name != "vertex") {
            // Element before vertices: skip (raro).
            if (has_list) 
                return fail("elemento con liste prima dei vertici: non supportato");
            if (fmt == 0) {
                std::string tmp;
                for (size_t i = 0; i < e.count; ++i) std::getline(f, tmp);
            } else {
                f.seekg(static_cast<std::streamoff>(row * e.count), std::ios::cur);
            }
            continue;
        }

        if (has_list) 
            return fail("liste nei vertici: non supportato");

        int ix = -1, iy = -1, iz = -1;
        std::vector<size_t> off(e.props.size());
        size_t o = 0;
        for (size_t k = 0; k < e.props.size(); ++k) {
            off[k] = o;
            o += e.props[k].size;
            if (e.props[k].name == "x") ix = static_cast<int>(k);
            if (e.props[k].name == "y") iy = static_cast<int>(k);
            if (e.props[k].name == "z") iz = static_cast<int>(k);
        }
        if (ix < 0 || iy < 0 || iz < 0) return fail("i vertici non hanno x, y, z");

        pc.reserve(e.count);

        if (fmt == 1) {
            const size_t CH = 65536;
            std::vector<unsigned char> buf(row * CH);
            for (size_t done = 0; done < e.count;) {
                size_t m = std::min(CH, e.count - done);
                f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(row * m));
                if (!f) return fail("file troncato");
                for (size_t r = 0; r < m; ++r) {
                    const unsigned char* b = buf.data() + r * row;
                    float x = static_cast<float>(read_scalar(b + off[ix], e.props[ix].size, e.props[ix].kind));
                    float y = static_cast<float>(read_scalar(b + off[iy], e.props[iy].size, e.props[iy].kind));
                    float z = static_cast<float>(read_scalar(b + off[iz], e.props[iz].size, e.props[iz].kind));
                    pc.push_back(x, y, z);
                }
                done += m;
            }
        } else {
            std::string l;
            for (size_t i = 0; i < e.count; ++i) {
                if (!std::getline(f, l)) return fail("file troncato");
                const char* s = l.c_str();
                char* end = nullptr;
                double v[3] = {0, 0, 0};
                for (size_t k = 0; k < e.props.size(); ++k) {
                    double val = std::strtod(s, &end);
                    if (end == s) return fail("riga ascii non valida");
                    s = end;
                    if (static_cast<int>(k) == ix) v[0] = val;
                    if (static_cast<int>(k) == iy) v[1] = val;
                    if (static_cast<int>(k) == iz) v[2] = val;
                }
                pc.push_back(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]));
            }
        }
        return true;  // other elements (faces, ...) are not needed
    }
    return fail("nessun elemento 'vertex'");
}

bool load_kitti_bin(const std::string& path, PointCloud& pc, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = path + ": " + m;
        return false;
    };
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return fail("impossibile aprire il file");
    const std::streamoff bytes = f.tellg();
    if (bytes < 0 || bytes % 16 != 0) return fail("dimensione non multipla di 16 byte");
    const size_t n = static_cast<size_t>(bytes) / 16;
    f.seekg(0);

    pc.clear();
    pc.reserve(n);
    const size_t CH = 65536;
    std::vector<float> buf(4 * CH);
    for (size_t done = 0; done < n;) {
        size_t m = std::min(CH, n - done);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(16 * m));
        if (!f) return fail("file troncato");
        for (size_t i = 0; i < m; ++i) pc.push_back(buf[4 * i], buf[4 * i + 1], buf[4 * i + 2]);
        done += m;
    }
    return true;  // la reflectance (4th column) discarded
}

bool load_cloud(const std::string& path, PointCloud& pc, std::string* err) {
    auto ends_with = [&](const char* ext) {
        const std::string e(ext);
        return path.size() >= e.size() && path.compare(path.size() - e.size(), e.size(), e) == 0;
    };
    if (ends_with(".ply")) return load_ply(path, pc, err);
    if (ends_with(".bin")) return load_kitti_bin(path, pc, err);
    if (err) *err = path + ": estensione non riconosciuta (usa .ply o .bin)";
    return false;
}

bool save_ply(const std::string& path, const PointCloud& pc, std::string* err) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        if (err) *err = path + ": impossibile scrivere il file";
        return false;
    }
    f << "ply\nformat binary_little_endian 1.0\n"
      << "element vertex " << pc.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\nend_header\n";
    const size_t CH = 65536;
    std::vector<float> buf(3 * CH);
    for (size_t done = 0; done < pc.size();) {
        size_t m = std::min(CH, pc.size() - done);
        for (size_t i = 0; i < m; ++i) {
            buf[3 * i] = pc.x[done + i];
            buf[3 * i + 1] = pc.y[done + i];
            buf[3 * i + 2] = pc.z[done + i];
        }
        f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(12 * m));
        done += m;
    }
    return static_cast<bool>(f);
}
