import React, { useEffect, useState } from "react";

const Modal = ({ active, handleModal, token, id, setErrorMessage }) => {
  const [pengaduan, setPengaduan] = useState("");
  const [pelapor, setPelapor] = useState("");
  const [kategori, setKategori] = useState("");
  
  
  useEffect(() => {
    const getData = async () => {
      const requestOptions = {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          // Authorization: "Bearer " + token,
        },
      };
      const response = await fetch(`/data/${id}`, requestOptions);

      if (!response.ok) {
        setErrorMessage("Could not get the data");
      } else {
        const data = await response.json();
        console.log(data)
        setPengaduan(data.data.tweets);
        setPelapor(data.data.users);
        setKategori(data.data.type);
      }
    };

    if (id) {
      getData();
    }
  }, [id, token]);

  const cleanFormData = () => {
    setPengaduan("");
    setPelapor("");
    setKategori("");
  };

  const handleCreateLead = async (e) => {
    e.preventDefault();
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Authorization: "Bearer " + token,
      },
      body: JSON.stringify({
    
        tweets: pengaduan,
        users: pelapor,
        type: kategori,
        mark: 'unprocessed',
        label: 'unlabelled'
      }),
    };
    const response = await fetch("/data", requestOptions);
    console.log(requestOptions)
    if (!response.ok) {
      setErrorMessage("Something went wrong when creating data");
    } else {
      cleanFormData();
      handleModal();
    }
  };

  const handleUpdateLead = async (e) => {
    e.preventDefault();
    const requestOptions = {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        // Authorization: "Bearer " + token,
      },
      body: JSON.stringify({
        type: kategori
      }),
    };
    const response = await fetch(`/data/${id}?type=${encodeURIComponent(kategori)}`, requestOptions);
    console.log(requestOptions)
    if (!response.ok) {
      setErrorMessage("Something went wrong when updating data");
      console.log(response)
    } else {
      cleanFormData();
      handleModal();
    }
  };

  return (
    <div className={`modal ${active && "is-active"}`}>
      <div className="modal-background" onClick={handleModal}></div>
      <div className="modal-card">
        <header className="modal-card-head has-background-primary-light">
          <h1 className="modal-card-title">
            {id ? "Update Data" : "Create Data"}
          </h1>
        </header>
        <section className="modal-card-body">
          <form>
            <div className="field">
              <label className="label">Pengaduan</label>
              <div className="control">
                <input
                  type="text"
                  placeholder="Enter first name"
                  value={pengaduan}
                  onChange={(e) => setPengaduan(e.target.value)}
                  className="input"
                  required
                />
              </div>
            </div>
            <div className="field">
              <label className="label">Pelapor</label>
              <div className="control">
                <input
                  type="text"
                  placeholder="Enter last name"
                  value={pelapor}
                  onChange={(e) => setPelapor(e.target.value)}
                  className="input"
                  required
                />
              </div>
            </div>
            <div className="field">
              <label className="label">Kategori Pengaduan</label>
              <div className="control">
                <input
                  type="text"
                  placeholder="Masukkan Kategori Pengaduan"
                  value={kategori}
                  onChange={(e) => setKategori(e.target.value)}
                  className="input"
                />
              </div>
            </div>
          </form>
        </section>
        <footer className="modal-card-foot has-background-primary-light">
          {id ? (
            <button className="button is-info" onClick={handleUpdateLead}>
              Update
            </button>
          ) : (
            <button className="button is-primary" onClick={handleCreateLead}>
              Create
            </button>
          )}
          <button className="button" onClick={handleModal}>
            Cancel
          </button>
        </footer>
      </div>
    </div>
  );
};

export default Modal;