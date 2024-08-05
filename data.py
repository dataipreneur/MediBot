{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNCAjwVzbla86GbpnX1Q7U4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/dataipreneur/MediBot/blob/main/data.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "moqefPNZVObr",
        "outputId": "e11e28a7-95ed-4658-aaa2-e7cf5060b586"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'MedPix-2.0'...\n",
            "remote: Enumerating objects: 2169, done.\u001b[K\n",
            "remote: Counting objects: 100% (2169/2169), done.\u001b[K\n",
            "remote: Compressing objects: 100% (2168/2168), done.\u001b[K\n",
            "remote: Total 2169 (delta 70), reused 0 (delta 0), pack-reused 0\u001b[K\n",
            "Receiving objects: 100% (2169/2169), 308.16 MiB | 25.73 MiB/s, done.\n",
            "Resolving deltas: 100% (70/70), done.\n",
            "Updating files: 100% (2065/2065), done.\n",
            "/content/MedPix-2.0\n",
            "/content/MedPix-2.0/splitted_dataset\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "267"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import json\n",
        "\n",
        "##Data modeliing\n",
        "\n",
        "!git clone https://github.com/CHILab1/MedPix-2.0.git\n",
        "%cd MedPix-2.0\n",
        "%cd splitted_dataset/\n",
        "\n",
        "#Create a data file\n",
        "# Read the JSON Lines file line by line\n",
        "data1 = []\n",
        "with open('data_train_1.jsonl', 'r') as f:\n",
        "    for line in f:\n",
        "        data1.append(pd.json_normalize(json.loads(line)))\n",
        "\n",
        "\n",
        "# Concatenate the list of dataframes into a single dataframe\n",
        "df1 = pd.concat(data1, ignore_index=True)\n",
        "\n",
        "l#en(df1['U_id'].unique())"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Read the JSON Lines file line by line\n",
        "data2 = []\n",
        "with open('descriptions_train_1.jsonl', 'r') as f:\n",
        "    for line in f:\n",
        "        data2.append(pd.json_normalize(json.loads(line)))\n",
        "\n",
        "# Concatenate the list of dataframes into a single dataframe\n",
        "df2 = pd.concat(data2, ignore_index=True)\n",
        "\n",
        "#print(df2)\n",
        "\n",
        "#len(df2['U_id'].unique())"
      ],
      "metadata": {
        "id": "8NEvD-zYVxJz"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data = pd.merge(df1, df2, on='U_id')\n",
        "#train_data.columns"
      ],
      "metadata": {
        "id": "XdjPr_H3V3JL"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "clean_data = train_data.loc[:, train_data.isna().any() == False]\n",
        "clean_data['image_path'] = clean_data['image'].apply(lambda x: '/content/drive/MyDrive/Falcon_Healthcare_agent/MedPix-2.0/images/' + x + '.png')\n",
        "clean_data['image_path']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RzGdm7QWV8_n",
        "outputId": "be806f4b-3b67-46f3-e179-8ce571382ced"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-8-b86484b7c74f>:2: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  clean_data['image_path'] = clean_data['image'].apply(lambda x: '/content/drive/MyDrive/Falcon_Healthcare_agent/MedPix-2.0/images/' + x + '.png')\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0      /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "1      /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "2      /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "3      /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "4      /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "                             ...                        \n",
              "805    /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "806    /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "807    /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "808    /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "809    /content/drive/MyDrive/Falcon_Healthcare_agent...\n",
              "Name: image_path, Length: 810, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Sampling\n",
        "\n",
        "image_list = list(clean_data['image_path'][0:10])\n",
        "\n",
        "sample_10 = clean_data[0:10]\n",
        "\n",
        "sample_10.to_csv('sample_10_final.csv')\n",
        "\n",
        "print(sample_10.columns)\n",
        "\n",
        "sample_10 = sample_10[['U_id','Case.Case Diagnosis', 'Case.Title','Topic.Category', 'Topic.Disease Discussion',\n",
        "       'Topic.Title', 'Type', 'Location', 'Location Category',\n",
        "       'Description.ACR Codes', 'Description.Age', 'Description.Caption',\n",
        "       'Description.Modality', 'Description.Sex']]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GvdA7umfWDC6",
        "outputId": "fe2fa603-1c34-4dfd-c944-a69a78d4619a"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['U_id', 'TAC', 'MRI', 'Case.Case Diagnosis', 'Case.Title',\n",
            "       'Topic.ACR Code', 'Topic.Category', 'Topic.Disease Discussion',\n",
            "       'Topic.Title', 'Type', 'image', 'Location', 'Location Category',\n",
            "       'Description.ACR Codes', 'Description.Age', 'Description.Caption',\n",
            "       'Description.Modality', 'Description.Sex', 'image_path'],\n",
            "      dtype='object')\n"
          ]
        }
      ]
    }
  ]
}