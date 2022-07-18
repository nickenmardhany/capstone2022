import React, { useContext, useEffect, useState } from "react";
import Modal from "./Modal";

import ErrorMessage from "./ErrorMessage";

import { UserContext } from "../context/UserContext";


const Table = () => {
  const [token] = useContext(UserContext);
  const [listdata, setListData] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [activeModal, setActiveModal] = useState(false);
  const [id, setId] = useState(null);

  const handleUpdate = async (id) => {
    setId(id);
    setActiveModal(true);
  };

  const handleDelete = async (id) => {

    const requestOptions = {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,

      },
    };
    const response = await fetch(`/data/${id}`, requestOptions);
    if (!response.ok) {
      setErrorMessage("Failed to delete lead");
    }

    getData();
  };

 
  
  const getData = async () => {
    
    const requestOptions = {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,

      },
    };
    const response = await fetch("/data", requestOptions);
    if (!response.ok) {
      setErrorMessage("Something went wrong. Couldn't load the leads");
    } else {
      const listdata = await response.json();
      setListData(listdata.data);
      console.log(listdata.data);
      setLoaded(true);
    }
  };

  useEffect(() => {
    getData();
  }, []);

  const handleModal = () => {
    setActiveModal(!activeModal);
    getData();
    setId(null);
  };

  return (
    <>
      <Modal
        active={activeModal}
        handleModal={handleModal}
        token={token}
        id={id}
        setErrorMessage={setErrorMessage}
      />
      {/* <button
        className="button is-fullwidth mb-5 is-info"
        onClick={() => <Register/>}
      >
        Tambahkan Akun Admin Baru
      </button> */}
      <button
        className="button is-fullwidth mb-5 is-primary"
        onClick={() => setActiveModal(true)}
      >
        Tambahkan Data Pengaduan Manual
      </button>
      
      
      <ErrorMessage message={errorMessage} />
      {loaded && listdata ? (
        <table className="table is-fullwidth is-striped is-hoverable is-narrow">
          <thead>
            <tr>
              <th>Pengaduan</th>
              <th>Nama Pelapor</th>
              <th>Label</th>
              <th>Kategori Pengaduan</th>
              <th>Aksi</th>
            </tr>
          </thead>
          <tbody>
            {listdata.map((data) => (
              <tr key={data.id}>
                <td>{data.tweets}</td>
                <td>{data.user}</td>
                <td>{data.label}</td>
                <td>{data.category}</td>
                
                <td>
                  <button
                    className="button mr-2 is-info is-light"
                    onClick={() => handleUpdate(data.id)}
                  >
                    Kategorisasi
                  </button>
                  <button
                    className="button mr-2 is-danger is-light"
                    onClick={() => handleDelete(data.id)}
                  >
                    Hapus Data
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>Loading</p>
      )}
    </>
  );
};

export default Table;