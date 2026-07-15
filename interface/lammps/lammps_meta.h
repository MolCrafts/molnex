/* Tiny meta.json field reader for pair_style molnex.
 *
 * Mirrors the deliberately-dependency-free string-scanning approach in
 * interface/src/model_runner.cpp (parse_device_field): meta.json is small and
 * flat enough that a real JSON library is not worth pulling into the LAMMPS
 * plugin link. We only read a handful of scalar fields from the `lammps` block
 * that molix.engine.export_for_lammps writes; key names are unique within the
 * file so a whole-text search is unambiguous.
 *
 * Header-only so the plugin links nothing extra beyond libmolnex_interface.
 */

#ifndef MOLNEX_LAMMPS_META_H
#define MOLNEX_LAMMPS_META_H

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace molnex::lammps_meta {

inline std::string read_text(const std::string& path) {
  std::ifstream is(path);
  if (!is) throw std::runtime_error("cannot open meta.json: " + path);
  std::stringstream ss;
  ss << is.rdbuf();
  return ss.str();
}

/* Position just past the colon following "key", or npos if the key is absent. */
inline std::size_t value_pos(const std::string& json, const std::string& key) {
  const std::string quoted = "\"" + key + "\"";
  auto k = json.find(quoted);
  if (k == std::string::npos) return std::string::npos;
  auto colon = json.find(':', k + quoted.size());
  if (colon == std::string::npos) return std::string::npos;
  return colon + 1;
}

inline bool has_field(const std::string& json, const std::string& key) {
  return value_pos(json, key) != std::string::npos;
}

inline std::string get_string(const std::string& json, const std::string& key) {
  auto v = value_pos(json, key);
  if (v == std::string::npos) throw std::runtime_error("meta.json: missing field '" + key + "'");
  auto open_q = json.find('"', v);
  if (open_q == std::string::npos) throw std::runtime_error("meta.json: '" + key + "' not a string");
  auto close_q = json.find('"', open_q + 1);
  if (close_q == std::string::npos)
    throw std::runtime_error("meta.json: '" + key + "' string not terminated");
  return json.substr(open_q + 1, close_q - open_q - 1);
}

inline double get_number(const std::string& json, const std::string& key) {
  auto v = value_pos(json, key);
  if (v == std::string::npos) throw std::runtime_error("meta.json: missing field '" + key + "'");
  try {
    return std::stod(json.substr(v));
  } catch (const std::exception&) {
    throw std::runtime_error("meta.json: '" + key + "' is not a number");
  }
}

inline bool get_bool(const std::string& json, const std::string& key, bool fallback) {
  auto v = value_pos(json, key);
  if (v == std::string::npos) return fallback;
  // skip whitespace, then look for 't'(rue)/'f'(alse)
  while (v < json.size() && (json[v] == ' ' || json[v] == '\t' || json[v] == '\n')) ++v;
  if (v < json.size() && (json[v] == 't' || json[v] == 'T')) return true;
  if (v < json.size() && (json[v] == 'f' || json[v] == 'F')) return false;
  return fallback;
}

}  // namespace molnex::lammps_meta

#endif  // MOLNEX_LAMMPS_META_H
