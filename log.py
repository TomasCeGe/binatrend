#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri/dev/binatrend/data/log.py
# Project: /Users/tomasmuniesa/Documents/develop/CeGe/Clientes/ascri
# Created Date: Friday, October 17th 2025, 9:55:19 am
# Author: Tomás
# -----
# Last Modified:  Friday, 17th October 2025 7:32:51 pm
# Modified By: Tomás (tomas@cege.es>)
# -----
# Copyright (c) 2025 Nousmedis, CeGe
#
# This product is licensed by CeGe.
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import logging
from colorlog import ColoredFormatter


def setup_logger_with_colors(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # === FORMATO PARA ARCHIVO (SIN COLORES) ===
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # === FORMATO PARA CONSOLA (CON COLORES) ===
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red'
        }
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger
