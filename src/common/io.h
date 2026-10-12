#pragma once

#ifndef IO_H
#define IO_H   

#include <string>
#include "PointCloud.h"

bool load_ply(const std::string& path, PointCloud& pc, std::string* err = nullptr);

bool load_kitti_bin(const std::string& path, PointCloud& pc, std::string* err = nullptr);

bool load_cloud(const std::string& path, PointCloud& pc, std::string* err = nullptr);

bool save_ply(const std::string& path, const PointCloud& pc, std::string* err); 


#endif