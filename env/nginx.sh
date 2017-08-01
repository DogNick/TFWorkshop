#!/bin/bash
#systemctl is-enabled nginx.service #查询nginx是否开机启动
#systemctl enable nginx.service #开机运行nginx
#systemctl disable nginx.service #取消开机运行nginx
#systemctl start nginx.service #启动nginx
#systemctl stop nginx.service #停止nginx
#systemctl restart nginx.service #重启nginx
#systemctl reload nginx.service #重新加载nginx配置文件
#systemctl status nginx.service #查询nginx运行状态
#systemctl --failed #显示启动失败的服务

#systemctl start nginx.service
systemctl restart nginx.service
#systemctl reload nginx.service
